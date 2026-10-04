"""Persistent single-seller desk; SQLite transactions protect conversation updates."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import uuid

from scripts.common import now, state_dir

DB_PATH = Path(os.environ.get('FLOOR_DESK_DB', str(state_dir() / 'floor.sqlite3')))


@contextmanager
def database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('BEGIN IMMEDIATE')
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize():
    with database() as db:
        db.execute('CREATE TABLE IF NOT EXISTS listing (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS inquiries (id TEXT PRIMARY KEY, body TEXT NOT NULL)')
        if db.execute('SELECT 1 FROM listing').fetchone():
            return
        listing = dict(item='Studio headphones', currency='KES', asking=12000, minimum=10000,
            condition='Good condition, lightly used. Includes the original cable and carrying case.',
            collection_location='Westlands, Nairobi', delivery_cost=500, availability='available',
            version=1, example=True, updated_at=now())
        db.execute('INSERT INTO listing VALUES (1, ?)', (json.dumps(listing),))
        examples = [
            ('Maya', 'Instagram', 'Hi! Are these still available? What condition are they in?'),
            ('Kevin', 'WhatsApp', 'I can do KES 10,000 cash and collect today. Would that work?'),
            ('Nia', 'Marketplace', 'Would you take KES 8,000? I can collect and pay in full.'),
            ('Sam', 'WhatsApp', 'Can you do KES 11,000 including delivery? The courier costs KES 500.'),
            ('Ali', 'Marketplace', 'I can pay KES 6,000 today and the remaining KES 5,000 on payday.'),
        ]
        for buyer, channel, text in examples:
            thread = new_thread(buyer, channel, text, example=True)
            db.execute('INSERT INTO inquiries VALUES (?, ?)', (thread['id'], json.dumps(thread)))


def new_thread(buyer_name, channel, text, example=False):
    stamp = now()
    return dict(id=uuid.uuid4().hex, buyer_name=buyer_name, channel=channel, example=example,
        status='open', version=1, messages=[dict(id=uuid.uuid4().hex, role='buyer', text=text, created_at=stamp)],
        analysis=None, draft=None, created_at=stamp, updated_at=stamp)


def desk():
    with database() as db:
        listing = json.loads(db.execute('SELECT body FROM listing WHERE id=1').fetchone()['body'])
        threads = [json.loads(row['body']) for row in db.execute('SELECT body FROM inquiries ORDER BY rowid')]
    return dict(listing=listing, inquiries=threads)


def inquiry(inquiry_id):
    with database() as db:
        row = db.execute('SELECT body FROM inquiries WHERE id=?', (inquiry_id,)).fetchone()
        if row is None:
            raise KeyError(inquiry_id)
        return json.loads(row['body'])


def create(buyer_name, channel, text):
    thread = new_thread(buyer_name, channel, text)
    with database() as db:
        db.execute('INSERT INTO inquiries VALUES (?, ?)', (thread['id'], json.dumps(thread)))
    return thread


def update(inquiry_id, operation, expected_version=None, expected_listing_version=None):
    with database() as db:
        row = db.execute('SELECT body FROM inquiries WHERE id=?', (inquiry_id,)).fetchone()
        if row is None:
            raise KeyError(inquiry_id)
        thread = json.loads(row['body'])
        if expected_version is not None and thread['version'] != expected_version:
            raise ValueError('The conversation changed. Review the latest message and try again.')
        if expected_listing_version is not None:
            listing = json.loads(db.execute('SELECT body FROM listing WHERE id=1').fetchone()['body'])
            if listing['version'] != expected_listing_version:
                raise ValueError('Your listing changed. Review the inquiry again with the new terms.')
        operation(thread)
        thread['version'] += 1
        thread['updated_at'] = now()
        db.execute('UPDATE inquiries SET body=? WHERE id=?', (json.dumps(thread), inquiry_id))
        return thread


def save_listing(values):
    with database() as db:
        current = json.loads(db.execute('SELECT body FROM listing WHERE id=1').fetchone()['body'])
        listing = dict(values, version=current['version'] + 1, example=False, updated_at=now())
        db.execute('UPDATE listing SET body=? WHERE id=1', (json.dumps(listing),))
        for row in db.execute('SELECT id, body FROM inquiries').fetchall():
            thread = json.loads(row['body'])
            thread.update(analysis=None, draft=None, version=thread['version'] + 1, updated_at=now())
            if thread['status'] == 'draft':
                thread['status'] = 'open'
            db.execute('UPDATE inquiries SET body=? WHERE id=?', (json.dumps(thread), row['id']))
    return desk()
