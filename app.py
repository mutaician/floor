"""Floor's local seller desk. Run .venv/bin/uvicorn app:app --port 8000."""

from collections import OrderedDict
from contextlib import asynccontextmanager
import hashlib
import base64
import json
import os
import re
import secrets
from threading import Lock
import time
import urllib.request
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from typing import Literal

from floor import context, proposal_from_interpretation
from scripts.common import ROOT, now, setup, state_dir, write_json
setup()  # Load .env before selecting the database location and authentication.
from scripts import desk_store as store


def selected_model():
    path = os.environ.get('TINKER_MODEL_PATH', '').strip()
    if path:
        return dict(sampler_path=path, prompt='strong')
    metadata = ROOT / 'results/selected_model.json'
    if metadata.exists():
        return json.loads(metadata.read_text())
    raise RuntimeError('Configure TINKER_MODEL_PATH on the server to analyze inquiries.')


class Listing(BaseModel):
    item: str = Field(min_length=1, max_length=120)
    currency: Literal['KES', 'USD'] = 'KES'
    asking: int = Field(strict=True, gt=0, le=100_000_000)
    minimum: int = Field(strict=True, gt=0, le=100_000_000)

    @model_validator(mode='after')
    def check_prices(self):
        self.item = self.item.strip()
        if not self.item or self.minimum > self.asking:
            raise ValueError('Use a minimum no higher than your asking price.')
        return self


class OfferRequest(Listing):
    buyer_message: str = Field(min_length=1, max_length=2000)

    @model_validator(mode='after')
    def check_message(self):
        self.buyer_message = self.buyer_message.strip()
        if not self.buyer_message:
            raise ValueError('Paste a buyer message first.')
        return self


class Interpretation(BaseModel):
    offer: int | None = Field(default=None, strict=True, gt=0, le=1_000_000_000)
    payment: Literal['full', 'deposit', 'installments', 'unclear']
    cost: int | None = Field(default=None, strict=True, ge=0, le=1_000_000_000)
    intent: Literal['offer', 'question', 'decline', 'other']


class RepriceRequest(Listing):
    interpretation: Interpretation


class SellerListing(Listing):
    condition: str = Field(default='', max_length=1000)
    collection_location: str = Field(default='', max_length=120)
    delivery_cost: int | None = Field(default=None, strict=True, ge=0, le=100_000_000)
    availability: Literal['available', 'reserved', 'sold'] = 'available'


class NewInquiry(BaseModel):
    buyer_name: str = Field(min_length=1, max_length=80)
    channel: Literal['WhatsApp', 'Instagram', 'Marketplace', 'Other'] = 'Marketplace'
    text: str = Field(min_length=1, max_length=2000)

    @model_validator(mode='after')
    def strip_text(self):
        self.buyer_name, self.text = self.buyer_name.strip(), self.text.strip()
        if not self.buyer_name or not self.text:
            raise ValueError('Add the buyer name and message.')
        return self


class BuyerMessage(BaseModel):
    text: str = Field(min_length=1, max_length=2000)

    @model_validator(mode='after')
    def strip_text(self):
        self.text = self.text.strip()
        if not self.text:
            raise ValueError('Add the buyer message.')
        return self


class DraftUpdate(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    version: int


class SentReply(DraftUpdate):
    terms_confirmed: Literal[True]


class TermUpdate(BaseModel):
    interpretation: Interpretation
    version: int


class StatusUpdate(BaseModel):
    status: Literal['closed', 'open']


def proposal(listing, interpreted):
    result = proposal_from_interpretation(listing, interpreted)
    action = result['action']
    if action == 'clarify':
        if interpreted['intent'] == 'question' or interpreted['offer'] is None:
            reason = 'There isn’t a clear total offer yet. Ask for the missing details.'
        elif interpreted['payment'] != 'full':
            reason = 'This payment isn’t confirmed in full. Clarify the terms before agreeing.'
        else:
            reason = 'Delivery costs are unresolved. Confirm who pays before settling on a price.'
    elif action == 'decline':
        reason = 'The buyer is passing. You can close the conversation politely.'
    elif action == 'accept':
        reason = 'The offer meets your asking price after the delivery costs shown below.'
    else:
        reason = 'This counter keeps you within your price boundary after the delivery costs shown below.'
    return dict(interpretation=interpreted, action=action, price=result['price'],
        reply=result['reply'], reason=reason, currency=listing['currency'])


def agent_proposal(listing, thread, interpreted):
    """Use the model's interpretation, known listing facts and seller pricing tools."""
    interpreted = dict(interpreted)
    latest = next(message['text'] for message in reversed(thread['messages']) if message['role'] == 'buyer')
    text = latest.lower()
    delivery_requested = bool(re.search(r'\b(deliver\w*|ship\w*|courier|postage)\b', text))
    used_delivery = False
    if delivery_requested and interpreted['intent'] == 'offer' and interpreted['cost'] is None and listing['delivery_cost'] is not None:
        interpreted['cost'] = listing['delivery_cost']
        used_delivery = True
    result = proposal(listing, interpreted)
    previous_quote = next((message for message in reversed(thread['messages'])
        if message['role']=='seller' and message.get('quoted_net') is not None), None)
    if (previous_quote and interpreted['intent']=='offer' and interpreted['payment']=='full'
        and interpreted['offer'] is not None and interpreted['cost'] is not None
        and interpreted['offer']-interpreted['cost'] >= max(listing['minimum'], previous_quote['quoted_net'])):
        result.update(action='accept', price=interpreted['offer'],
            reason='The buyer meets the price you last offered, after delivery costs. You don’t need to counter again.')
    facts, missing = [], []
    asks_availability = bool(re.search(r'\b(available|still for sale)\b', text))
    asks_condition = bool(re.search(r'\b(condition|scratches|damage|damaged|working|accessories|included)\b', text))
    asks_collection = bool(re.search(r'\b(where|location|collect from|pick.?up from)\b', text))
    asks_price = interpreted['intent'] == 'question' and bool(re.search(r'\b(price|best|lowest|last|discount)\b', text))
    if asks_availability:
        facts.append({'available':'Yes, it’s available.', 'reserved':'It’s currently reserved.', 'sold':'It’s already sold.'}[listing['availability']])
    if asks_condition:
        if listing['condition'].strip(): facts.append(listing['condition'].strip())
        else: missing.append('Add the item’s condition to your listing.')
    if asks_collection:
        if listing['collection_location'].strip(): facts.append('Collection is from ' + listing['collection_location'].strip().rstrip('.') + '.')
        else: missing.append('Add a collection location to your listing.')
    if listing['availability'] != 'available':
        result.update(action='close', price=None,
            reply='Thanks for your interest. ' + ('This item is already sold.' if listing['availability']=='sold' else 'This item is currently reserved.'),
            reason='Your saved listing is ' + listing['availability'] + '. Let the buyer know before negotiating.')
    elif interpreted['intent'] == 'question' and (facts or asks_price) and not missing:
        if asks_price:
            facts.append(f"The asking price is {listing['currency']} {listing['asking']:,}, with full payment.")
        result.update(action='answer', price=listing['asking'] if asks_price else None, reply=' '.join(facts),
            reason='Answer the buyer’s question using your saved listing details.')
    elif missing and interpreted['intent'] == 'question':
        result.update(action='needs_info', price=None, reply='Let me confirm those details before we agree.',
            reason='The buyer asked for information that isn’t in your listing yet.')
    elif result['action'] in {'accept','counter'}:
        verb='accept' if result['action']=='accept' else 'do'
        result['reply']=f"I can {verb} {listing['currency']} {result['price']:,} total, with full payment."
        if delivery_requested and interpreted['cost'] and interpreted['cost'] > 0:
            result['reply'] += ' That includes the delivery cost.'
        if facts: result['reply'] += ' ' + ' '.join(facts)
    elif result['action']=='clarify' and interpreted['payment'] in {'deposit','installments'}:
        result['reply']='Thanks for the offer. I’m looking for full payment. What total could you offer on that basis?'
    if result['action']=='clarify':
        if interpreted['offer'] is None: missing.append('Confirm the total offer.')
        if interpreted['payment']!='full': missing.append('Confirm full payment.')
        if interpreted['cost'] is None: missing.append('Confirm who covers delivery and its cost.')
    payment_labels={'full':'Full payment', 'deposit':'Deposit only', 'installments':'Installments', 'unclear':'Payment unclear'}
    summary = dict(offer=interpreted['offer'], payment=payment_labels[interpreted['payment']],
        seller_cost=interpreted['cost'], message_type=interpreted['intent'])
    result.update(summary=summary, missing=missing, tool_steps=[
        dict(tool='Listing lookup', detail=listing['item'] + ' · ' + listing['availability']),
        dict(tool='Conversation read', detail=f"{len(thread['messages'])} saved message(s); latest buyer message interpreted with Tinker"),
        dict(tool='Seller terms checked', detail='Approved earlier quote and current terms checked.' if previous_quote else ('Saved delivery cost applied.' if used_delivery else 'Price, payment and delivery reviewed.')),
        dict(tool='Reply prepared', detail='Seller review required before recording a sent reply.')])
    return result


class FloorModel:
    def __init__(self):
        self.lock = Lock()
        self.service = self.sampler = self.renderer = None
        self.cache = OrderedDict()
        self.price_checked = 0

    def refresh_prices(self):
        path = state_dir() / 'prices.json'
        if time.time() - self.price_checked < 1800:
            return
        from datetime import datetime, timezone
        prices = json.loads(path.read_text()) if path.exists() else {}
        stamp = prices.get('checked_at_utc')
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds() if stamp else 99999
        if age > 1800:
            request = urllib.request.Request('https://tinker-docs.thinkingmachines.ai/tinker/models.json',
                headers={'User-Agent':'Floor-demo/0.1'})
            with urllib.request.urlopen(request, timeout=15) as response:
                catalog = json.load(response)
            write_json(path, dict(checked_at_utc=now(), catalog=catalog))
        self.price_checked = time.time()

    def __call__(self, listing):
        with self.lock:
            import tinker
            from tinker.lib.retry_handler import RetryConfig
            from tinker_cookbook.renderers import get_text_content
            from scripts.budget import reserve, finish, uncertain, locked_ledger
            from scripts.rendering import build_renderer, messages
            config = setup()
            selected = selected_model()
            fingerprint = hashlib.sha256(json.dumps([listing, selected['sampler_path']], sort_keys=True).encode()).hexdigest()
            if fingerprint in self.cache:
                self.cache.move_to_end(fingerprint)
                return self.cache[fingerprint]
            if self.sampler is None:
                self.renderer, _ = build_renderer(config)
                self.prompt = json.loads((ROOT / f"prompts/{selected['prompt']}.json").read_text())
                self.service = tinker.ServiceClient()
                self.sampler = self.service.create_sampling_client(model_path=selected['sampler_path'],
                    retry_config=RetryConfig(enable_retry_logic=False, progress_timeout=90))
                self.model_path = selected['sampler_path']
            if selected['sampler_path'] != self.model_path:
                raise RuntimeError('Restart the desk to load the newly selected model.')
            self.refresh_prices()
            prompt = self.prompt
            model_context = json.loads(context(listing))
            if 'conversation_history' in listing:
                model_context['conversation_history'] = listing['conversation_history']
                model_context['seller_information'] = {key:listing.get(key) for key in ('condition','collection_location','delivery_cost','availability')}
                prompt = dict(self.prompt, system=self.prompt['system'] + '\nInterpret the newest buyer_message. Conversation history is context, not a fresh offer: do not mistake an earlier seller price for the newest buyer offer. Carry unresolved payment and delivery terms forward unless the buyer changes them. Seller information contains known listing facts. Continue to return the same JSON schema.')
            rendered = self.renderer.build_generation_prompt(messages(prompt, json.dumps(model_context, ensure_ascii=False)))
            params = tinker.SamplingParams(max_tokens=160, temperature=0.2, top_p=0.9, seed=42,
                stop=self.renderer.get_stop_sequences())
            request_id = 'floor-demo:' + uuid.uuid4().hex
            reserve(config, request_id, rendered.length, 160, 'floor_demo')
            try:
                response = self.sampler.sample(rendered, num_samples=1, sampling_params=params).result(timeout=150)
                sequence = response.sequences[0]
                # Record charges without persisting buyer messages or model text.
                finish(request_id, dict(request_id=request_id, output_tokens=len(sequence.tokens), model=self.model_path))
                parsed, termination = self.renderer.parse_response(sequence.tokens)
                if str(termination) != 'stop_sequence':
                    raise RuntimeError('The response did not finish. Try a shorter buyer message.')
                content = json.loads(get_text_content(parsed).strip())
                interpreted = Interpretation.model_validate(content).model_dump()
                result = proposal(listing, interpreted)
                result['request_id'] = request_id
                self.cache[fingerprint] = result
                if len(self.cache) > 32:
                    self.cache.popitem(last=False)
                return result
            except Exception as error:
                with locked_ledger() as ledger:
                    status = next(row['status'] for row in ledger['requests'] if row['request_id'] == request_id)
                if status != 'complete':
                    uncertain(request_id, error)
                raise

    def close(self):
        if self.service:
            self.service.close('success').result(timeout=30)


model = FloorModel()


@asynccontextmanager
async def lifespan(application):
    if bool(os.environ.get('FLOOR_USERNAME')) != bool(os.environ.get('FLOOR_PASSWORD')):
        raise RuntimeError('Set both FLOOR_USERNAME and FLOOR_PASSWORD, or neither.')
    store.initialize()
    from scripts.budget import locked_ledger
    with locked_ledger():
        pass
    yield
    from starlette.concurrency import run_in_threadpool
    await run_in_threadpool(model.close)


app = FastAPI(title='Floor', docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')


@app.middleware('http')
async def protect_desk(request, call_next):
    username, password = os.environ.get('FLOOR_USERNAME'), os.environ.get('FLOOR_PASSWORD')
    if username and password and request.url.path != '/health':
        try:
            scheme, encoded = request.headers.get('authorization', '').split(' ', 1)
            if scheme.lower() != 'basic':
                raise ValueError('Expected basic authentication')
            supplied_user, supplied_password = base64.b64decode(encoded, validate=True).decode('utf-8').split(':', 1)
            valid_user = secrets.compare_digest(supplied_user.encode(), username.encode())
            valid_password = secrets.compare_digest(supplied_password.encode(), password.encode())
            if not (valid_user and valid_password):
                raise ValueError('Invalid credentials')
        except (ValueError, UnicodeError):
            return Response(status_code=401, headers={'WWW-Authenticate':'Basic realm="Floor", charset="UTF-8"'})
    return await call_next(request)


@app.get('/')
def index():
    return FileResponse(ROOT / 'static/index.html', headers={'Cache-Control':'no-store'})


@app.get('/health')
def health():
    return dict(status='ok', model_selected=bool(os.environ.get('TINKER_MODEL_PATH')) or (ROOT / 'results/selected_model.json').exists())


@app.post('/propose')
def propose_offer(request: OfferRequest):
    try:
        return model(request.model_dump())
    except Exception as error:
        safe = {'Budget stop threshold reached; no request dispatched.',
            'Configure TINKER_MODEL_PATH on the server to analyze inquiries.',
            'The response did not finish. Try a shorter buyer message.',
            'Restart the desk to load the newly selected model.'}
        message = str(error) if str(error) in safe else 'Floor couldn’t read that offer just now. Your message is still here; try again.'
        raise HTTPException(status_code=503, detail=message) from None


@app.post('/reprice')
def reprice_offer(request: RepriceRequest):
    return proposal(request.model_dump(exclude={'interpretation'}), request.interpretation.model_dump())


def get_thread(inquiry_id):
    try: return store.inquiry(inquiry_id)
    except KeyError: raise HTTPException(404, 'Inquiry not found.') from None


def save_thread(inquiry_id, operation, **kwargs):
    try: return store.update(inquiry_id, operation, **kwargs)
    except KeyError: raise HTTPException(404, 'Inquiry not found.') from None
    except ValueError as error: raise HTTPException(409, str(error)) from None


@app.get('/desk')
def seller_desk():
    return store.desk()


@app.put('/listing')
def save_seller_listing(request: SellerListing):
    return store.save_listing(request.model_dump())


@app.post('/inquiries')
def add_inquiry(request: NewInquiry):
    return store.create(request.buyer_name, request.channel, request.text)


@app.post('/inquiries/{inquiry_id}/messages')
def add_buyer_message(inquiry_id: str, request: BuyerMessage):
    def operation(thread):
        thread['messages'].append(dict(id=uuid.uuid4().hex, role='buyer', text=request.text, created_at=now()))
        thread.update(status='open', analysis=None, draft=None)
    return save_thread(inquiry_id, operation)


@app.post('/inquiries/{inquiry_id}/analyze')
def review_inquiry(inquiry_id: str):
    thread = get_thread(inquiry_id)
    listing = store.desk()['listing']
    if thread['status']=='closed' or thread['messages'][-1]['role']!='buyer':
        raise HTTPException(409, 'Add a new buyer message or reopen this inquiry before reviewing it.')
    history = []
    characters = 0
    for message in reversed(thread['messages'][:-1][-8:]):
        characters += len(message['text'])
        if characters > 6000: break
        history.insert(0, dict(role=message['role'], text=message['text']))
    payload = dict(listing, buyer_message=thread['messages'][-1]['text'], conversation_history=history)
    try:
        interpreted = model(payload)['interpretation']
    except Exception:
        raise HTTPException(503, 'Floor couldn’t review this inquiry just now. Your conversation is saved; try again.') from None
    analysis = agent_proposal(listing, thread, interpreted)
    def operation(current):
        current.update(analysis=analysis, draft=analysis['reply'], status='draft')
    return save_thread(inquiry_id, operation, expected_version=thread['version'], expected_listing_version=listing['version'])


@app.post('/inquiries/{inquiry_id}/terms')
def correct_inquiry_terms(inquiry_id: str, request: TermUpdate):
    thread = get_thread(inquiry_id)
    if thread['analysis'] is None or thread['status']!='draft':
        raise HTTPException(409, 'Review this inquiry before correcting its terms.')
    listing = store.desk()['listing']
    analysis = agent_proposal(listing, thread, request.interpretation.model_dump())
    def operation(current): current.update(analysis=analysis, draft=analysis['reply'])
    return save_thread(inquiry_id, operation, expected_version=request.version, expected_listing_version=listing['version'])


@app.post('/inquiries/{inquiry_id}/draft')
def save_inquiry_draft(inquiry_id: str, request: DraftUpdate):
    def operation(thread):
        if thread['status']!='draft' or thread['analysis'] is None:
            raise ValueError('Review the inquiry before saving a reply.')
        thread['draft']=request.text
    return save_thread(inquiry_id, operation, expected_version=request.version)


@app.post('/inquiries/{inquiry_id}/sent')
def record_sent_reply(inquiry_id: str, request: SentReply):
    def operation(thread):
        if thread['status']!='draft' or thread['analysis'] is None:
            raise ValueError('Review the inquiry before recording a sent reply.')
        analysis=thread['analysis']
        quoted_net=None
        if analysis['price'] is not None and analysis['interpretation']['cost'] is not None:
            amounts={int(amount.replace(',','')) for amount in re.findall(r'\b'+re.escape(analysis['currency'])+r'\s*(\d[\d,]*)',request.text)}
            if amounts=={analysis['price']}:
                quoted_net=analysis['price']-analysis['interpretation']['cost']
        thread['messages'].append(dict(id=uuid.uuid4().hex, role='seller', text=request.text, created_at=now(), quoted_net=quoted_net))
        thread.update(status='closed' if thread['analysis']['action'] in {'close','decline'} else 'waiting', draft=None)
    return save_thread(inquiry_id, operation, expected_version=request.version)


@app.post('/inquiries/{inquiry_id}/status')
def change_inquiry_status(inquiry_id: str, request: StatusUpdate):
    def operation(thread):
        thread['status']=request.status if request.status=='closed' or thread['messages'][-1]['role']=='buyer' else 'waiting'
        thread.update(analysis=None, draft=None)
    return save_thread(inquiry_id, operation)
