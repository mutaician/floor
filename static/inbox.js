'use strict';
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const format = value => new Intl.NumberFormat('en',{maximumFractionDigits:0}).format(value);
const money = value => `${deskData.listing.currency} ${format(value)}`;
let deskData = {listing:null,inquiries:[]}, activeId=null, filter='all', dirty=false, termsPending=false, savingDraft=false, draftTimer, termTimer;
const busy = new Set();
const current = () => deskData.inquiries.find(thread => thread.id === activeId);
const statusLabels={open:'Needs reply',draft:'Draft ready',waiting:'Waiting',closed:'Closed'};
async function api(path,body,method='POST') {
  const response=await fetch(path,{method,headers:{'Content-Type':'application/json'},...(body===undefined?{}:{body:JSON.stringify(body)})});
  const data=await response.json();
  if(!response.ok) throw new Error(typeof data.detail==='string'?data.detail:'Check the details and try again.');
  return data;
}
function showError(id,message){$(id).textContent=message;$(id).hidden=!message;}
function countsAndListing(){
  const listing=deskData.listing;if(!listing)return;
  $('listing-title').textContent=listing.item;
  $('listing-availability').textContent=listing.availability;
  $('listing-details').textContent=[listing.collection_location?'Collection: '+listing.collection_location:null,listing.delivery_cost===null?'Delivery cost unconfirmed':'Delivery: '+money(listing.delivery_cost),'Full payment'].filter(Boolean).join(' · ');
  $('listing-asking').textContent=money(listing.asking);$('listing-minimum').textContent=money(listing.minimum);
  $('needs-count').textContent=deskData.inquiries.filter(thread=>['open','draft'].includes(thread.status)).length;
  $('waiting-count').textContent=deskData.inquiries.filter(thread=>thread.status==='waiting').length;
  $('inquiry-count').textContent=deskData.inquiries.length;
  const examples=deskData.inquiries.filter(thread=>thread.example).length;
  $('sample-banner').hidden=!examples;
  $('sample-banner').textContent=examples?`${examples} sample inquiries are included. Add your own to try Floor with real messages.`:'';
}
function renderRows(){
  document.querySelectorAll('[data-filter]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.filter===filter)));
  const rows=deskData.inquiries.filter(thread=>filter==='all'||(filter==='needs'&&['open','draft'].includes(thread.status))||(filter==='waiting'&&thread.status==='waiting'));
  $('inquiry-list').innerHTML=rows.length?rows.map(thread=>{
    const latest=[...thread.messages].reverse().find(message=>message.role==='buyer');
    return `<button type="button" class="inquiry-row" data-id="${thread.id}" aria-current="${thread.id===activeId}"><span class="inquiry-row-top"><span class="avatar" aria-hidden="true">${escapeHTML(thread.buyer_name.slice(0,1).toUpperCase())}</span><span class="inquiry-row-name">${escapeHTML(thread.buyer_name)}</span>${thread.example?'<span class="sample-chip">SAMPLE</span>':''}</span><span class="inquiry-row-preview">${escapeHTML(latest?.text||'No messages yet.')}</span><span class="inquiry-row-bottom"><span>${escapeHTML(thread.channel)}</span><span class="status-badge status-${thread.status}">${statusLabels[thread.status]}</span></span></button>`;
  }).join(''):'<p class="empty-list">No inquiries in this view.</p>';
}
function renderConversation(){
  const thread=current();
  $('conversation-name').textContent=thread?.buyer_name||'Choose an inquiry';
  $('conversation-avatar').textContent=thread?.buyer_name.slice(0,1).toUpperCase()||'—';
  $('conversation-channel').textContent=thread?`${thread.channel} · ${statusLabels[thread.status]}${thread.example?' · Sample conversation':''}`:'';
  $('close-inquiry').hidden=!thread;
  $('close-inquiry').textContent=thread?.status==='closed'?'Reopen inquiry':'Close inquiry';
  $('sample-followup').hidden=!thread?.example;
  $('message-form').hidden=!thread;
  $('conversation-messages').innerHTML=thread?'<p class="conversation-note">'+(thread.example?'SAMPLE INQUIRY · FOR TRYING THE DESK':'SAVED BUYER CONVERSATION')+'</p>'+thread.messages.map(message=>{
    const time=new Date(message.created_at).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
    return `<article class="chat-message ${message.role}"><div class="chat-meta"><strong>${escapeHTML(message.role==='seller'?'You':thread.buyer_name)}</strong><span>${escapeHTML(time)}</span></div><p class="chat-text">${escapeHTML(message.text)}</p></article>`;
  }).join(''):'<p class="conversation-placeholder">Add a buyer inquiry to start.</p>';
  $('conversation-messages').scrollTop=$('conversation-messages').scrollHeight;
}
function enableReplyActions(){
  const thread=current(), blocked=!thread||thread.status!=='draft'||busy.has(activeId)||termsPending||savingDraft||!$('reply-text').value.trim();
  $('copy-reply').disabled=blocked||!$('confirmed').checked;
  $('mark-sent').disabled=blocked||!$('confirmed').checked;
  $('save-draft').disabled=blocked||!dirty;
}
function renderAgent(){
  const thread=current(), working=busy.has(activeId), analysis=thread?.status==='draft'?thread.analysis:null;
  const ready=!!thread&&thread.status!=='closed'&&thread.messages.at(-1)?.role==='buyer';
  $('analyze-inquiry').disabled=!ready||working;
  $('analyze-label').textContent=working?'Reviewing inquiry…':analysis?'Review latest message':'Prepare next reply';
  $('analyze-icon').hidden=working;document.querySelector('.spinner').hidden=!working;
  $('agent-loading').hidden=!working;$('agent-empty').hidden=working||!!analysis;$('agent-result').hidden=working||!analysis;
  $('agent-context').textContent=thread?`For ${thread.buyer_name}. Uses your saved listing and this conversation.`:'Choose an inquiry to prepare a reply.';
  if(!analysis){
    const waiting=thread?.status==='waiting', closed=thread?.status==='closed';
    document.querySelector('.agent-empty-title').innerHTML=waiting?'The reply is out.<br><em>Room to breathe.</em>':closed?'Conversation closed.<br><em>On your terms.</em>':'A steady hand.<br><em>Not the final word.</em>';
    document.querySelector('.agent-empty>p').textContent=waiting?'Add the buyer’s next message when they respond.':closed?'Reopen this inquiry if you want to continue.':'Review an inquiry to see Floor’s recommendation.';
    enableReplyActions();return;
  }
  const labels={counter:['SUGGESTED COUNTER','PRICE BOUNDARY'],accept:['OFFER WORKS','ON YOUR TERMS'],clarify:['CLARIFY THE TERMS','NEEDS AN ANSWER'],decline:['CLOSE THE CONVERSATION','KEEP IT SIMPLE'],close:['CLOSE THE CONVERSATION','LISTING STATUS'],answer:['ANSWER THE BUYER','LISTING FACTS'],needs_info:['ADD LISTING DETAILS','MISSING INFORMATION']};
  const [action,tag]=labels[analysis.action];$('action-label').textContent=action;$('decision-tag').textContent=tag;
  $('result-currency').textContent=analysis.price===null?'':analysis.currency;
  $('result-price').textContent=analysis.price!==null?format(analysis.price):({clarify:'Ask first.',decline:'Leave it there.',close:'Let them know.',answer:'Keep it clear.',needs_info:'Fill in the details.'}[analysis.action]);
  document.querySelector('.decision-price').classList.toggle('is-text',analysis.price===null);
  $('reason').textContent=analysis.reason;
  $('missing-terms').hidden=!analysis.missing.length;
  $('missing-terms').innerHTML=analysis.missing.map(message=>`<p>${escapeHTML(message)}</p>`).join('');
  $('review-offer').value=analysis.interpretation.offer??'';$('review-payment').value=analysis.interpretation.payment;
  $('review-cost').value=analysis.interpretation.cost??'';$('review-intent').value=analysis.interpretation.intent;
  $('reply-text').value=thread.draft??analysis.reply;$('reply-edited').textContent='SAVED DRAFT';
  $('confirmed').checked=false;$('copy-status').textContent='';$('review-status').textContent='';
  $('tool-steps').innerHTML=analysis.tool_steps.map(step=>`<li><strong>${escapeHTML(step.tool)}</strong><span>${escapeHTML(step.detail)}</span></li>`).join('');
  dirty=false;enableReplyActions();
}
function renderDesk(){countsAndListing();renderRows();renderConversation();renderAgent();}
function acceptThread(thread,{agent=true,conversation=true}={}){
  const index=deskData.inquiries.findIndex(row=>row.id===thread.id);
  if(index<0)deskData.inquiries.unshift(thread);else deskData.inquiries[index]=thread;
  countsAndListing();renderRows();
  if(thread.id===activeId){if(conversation)renderConversation();if(agent)renderAgent();}
}
async function saveDraft(silent=false){
  clearTimeout(draftTimer);
  if(savingDraft){draftTimer=setTimeout(()=>saveDraft(true),200);return false;}
  const thread=current();if(!thread||!dirty||thread.status!=='draft')return true;
  const text=$('reply-text').value;if(!text.trim())return false;
  const id=thread.id;savingDraft=true;enableReplyActions();
  try{
    const saved=await api(`/inquiries/${id}/draft`,{text,version:thread.version});
    acceptThread(saved,{agent:false,conversation:false});
    if(id===activeId){dirty=$('reply-text').value!==text;$('reply-edited').textContent=dirty?'UNSAVED EDIT':'SAVED DRAFT';if(!silent)$('copy-status').textContent='Draft saved.';if(dirty)draftTimer=setTimeout(()=>saveDraft(true),650);}
    return true;
  }catch(error){if(id===activeId){$('reply-edited').textContent='UNSAVED EDIT';$('copy-status').textContent=error.message;}return false;}
  finally{savingDraft=false;enableReplyActions();}
}
async function selectThread(id){
  if(id===activeId)return;
  if(dirty&&!await saveDraft(true))return;
  clearTimeout(termTimer);termsPending=false;dirty=false;activeId=id;
  localStorage.setItem('floor-active-inquiry',id);showError('agent-error','');showError('desk-error','');$('new-message').value='';renderRows();renderConversation();renderAgent();
  if(matchMedia('(max-width:660px)').matches)document.querySelector('.conversation-pane').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'instant':'smooth',block:'start'});
}
$('inquiry-list').addEventListener('click',event=>{const button=event.target.closest('[data-id]');if(button)selectThread(button.dataset.id);});
for(const button of document.querySelectorAll('[data-filter]'))button.addEventListener('click',()=>{filter=button.dataset.filter;renderRows();});
$('analyze-inquiry').addEventListener('click',async()=>{
  const thread=current();if(!thread||busy.has(thread.id))return;
  if(dirty&&!await saveDraft(true))return;
  const id=thread.id;busy.add(id);showError('agent-error','');renderAgent();
  try{
    const updated=await api(`/inquiries/${id}/analyze`,{});busy.delete(id);acceptThread(updated);
    if(id===activeId&&matchMedia('(max-width:1000px)').matches)document.querySelector('.agent-pane').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'instant':'smooth',block:'start'});
  }catch(error){busy.delete(id);if(id===activeId){renderAgent();showError('agent-error',error.message);}}
});
$('message-form').addEventListener('submit',async event=>{
  event.preventDefault();const thread=current(),text=$('new-message').value.trim();if(!thread||!text)return;
  const id=thread.id;$('add-message').disabled=true;clearTimeout(draftTimer);clearTimeout(termTimer);termsPending=false;
  try{const updated=await api(`/inquiries/${id}/messages`,{text});if(id===activeId){$('new-message').value='';dirty=false;}acceptThread(updated);showError('desk-error','');}
  catch(error){showError('desk-error',error.message);}finally{$('add-message').disabled=false;}
});
$('sample-followup').addEventListener('click',()=>{const amount=Math.ceil((deskData.listing.asking+deskData.listing.minimum)/2);$('new-message').value=`I can do ${money(amount)} in full and collect. Would that work?`;$('new-message').focus();});
$('close-inquiry').addEventListener('click',async()=>{
  const thread=current();if(!thread)return;
  clearTimeout(draftTimer);clearTimeout(termTimer);termsPending=false;
  try{acceptThread(await api(`/inquiries/${thread.id}/status`,{status:thread.status==='closed'?'open':'closed'}));showError('desk-error','');}
  catch(error){showError('desk-error',error.message);}
});
function correctTerms(){
  const thread=current();if(!thread?.analysis||thread.status!=='draft')return;
  const id=thread.id;clearTimeout(termTimer);clearTimeout(draftTimer);termsPending=true;$('confirmed').checked=false;$('review-status').textContent='Updating the reply…';enableReplyActions();
  termTimer=setTimeout(async()=>{
    const offer=$('review-offer').value.trim(),cost=$('review-cost').value.trim();
    const interpretation={offer:offer===''?null:Number(offer),cost:cost===''?null:Number(cost),payment:$('review-payment').value,intent:$('review-intent').value};
    if((interpretation.offer!==null&&(!Number.isSafeInteger(interpretation.offer)||interpretation.offer<=0))||(interpretation.cost!==null&&(!Number.isSafeInteger(interpretation.cost)||interpretation.cost<0))){$('review-status').textContent='Use whole positive amounts. Delivery can be zero.';return;}
    try{const updated=await api(`/inquiries/${id}/terms`,{interpretation,version:current().version});if(id===activeId)termsPending=false;acceptThread(updated);}
    catch(error){if(id===activeId)$('review-status').textContent=error.message;}
    finally{enableReplyActions();}
  },350);
}
document.querySelectorAll('.review-grid input,.review-grid select').forEach(input=>input.addEventListener('input',correctTerms));
$('reply-text').addEventListener('input',()=>{dirty=true;$('reply-edited').textContent='SAVING EDIT…';$('copy-status').textContent='';enableReplyActions();clearTimeout(draftTimer);draftTimer=setTimeout(()=>saveDraft(true),650);});
$('save-draft').addEventListener('click',()=>saveDraft());$('confirmed').addEventListener('change',enableReplyActions);
$('copy-reply').addEventListener('click',async()=>{
  if($('copy-reply').disabled)return;
  if(dirty&&!await saveDraft(true))return;
  try{await navigator.clipboard.writeText($('reply-text').value.trim());$('copy-status').textContent='Copied. Send it in your marketplace or chat.';}
  catch{$('reply-text').focus();$('reply-text').select();$('copy-status').textContent='Reply selected. Use your device’s copy command.';}
});
$('mark-sent').addEventListener('click',async()=>{
  if($('mark-sent').disabled)return;
  const thread=current(),text=$('reply-text').value.trim();clearTimeout(draftTimer);
  $('mark-sent').disabled=true;
  try{const updated=await api(`/inquiries/${thread.id}/sent`,{text,version:thread.version,terms_confirmed:true});dirty=false;acceptThread(updated);showError('desk-error','');}
  catch(error){showError('desk-error',error.message);enableReplyActions();}
});
$('edit-listing').addEventListener('click',()=>{
  const listing=deskData.listing;if(!listing)return;
  for(const field of ['item','currency','asking','minimum','condition','availability'])$('edit-'+field).value=listing[field];
  $('edit-collection').value=listing.collection_location;$('edit-delivery').value=listing.delivery_cost??'';showError('listing-error','');$('listing-dialog').showModal();
});
$('listing-form').addEventListener('submit',async event=>{
  event.preventDefault();const delivery=$('edit-delivery').value.trim();
  const listing={item:$('edit-item').value.trim(),currency:$('edit-currency').value,asking:Number($('edit-asking').value),minimum:Number($('edit-minimum').value),condition:$('edit-condition').value.trim(),availability:$('edit-availability').value,collection_location:$('edit-collection').value.trim(),delivery_cost:delivery===''?null:Number(delivery)};
  if(listing.minimum>listing.asking){showError('listing-error','The private floor must be no higher than your asking price.');return;}
  const button=$('listing-form').querySelector('[type=submit]');button.disabled=true;
  try{deskData=await api('/listing',listing,'PUT');clearTimeout(draftTimer);clearTimeout(termTimer);dirty=false;termsPending=false;$('listing-dialog').close();renderDesk();}
  catch(error){showError('listing-error',error.message);}finally{button.disabled=false;}
});
$('new-inquiry').addEventListener('click',()=>{$('inquiry-form').reset();showError('inquiry-error','');$('inquiry-dialog').showModal();});
$('inquiry-form').addEventListener('submit',async event=>{
  event.preventDefault();const data={buyer_name:$('buyer-name').value.trim(),channel:$('buyer-channel').value,text:$('first-message').value.trim()};
  const button=$('inquiry-form').querySelector('[type=submit]');button.disabled=true;
  try{const thread=await api('/inquiries',data);acceptThread(thread);filter='all';$('inquiry-dialog').close();await selectThread(thread.id);}
  catch(error){showError('inquiry-error',error.message);}finally{button.disabled=false;}
});
document.querySelectorAll('[data-close]').forEach(button=>button.addEventListener('click',()=>$(button.dataset.close).close()));
async function start(){
  try{deskData=await api('/desk',undefined,'GET');const saved=localStorage.getItem('floor-active-inquiry');activeId=deskData.inquiries.some(thread=>thread.id===saved)?saved:deskData.inquiries[0]?.id;renderDesk();}
  catch(error){showError('desk-error','The seller desk couldn’t load. '+error.message);$('analyze-inquiry').disabled=true;}
}
start();
