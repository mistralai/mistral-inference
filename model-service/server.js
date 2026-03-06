const express = require('express');
const fetch = require('node-fetch');
const dotenv = require('dotenv');

dotenv.config();

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 7071;
const OPENAI_BASE_URL = (process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1').replace(/\/+$/, '');
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
const MODEL_NAME = process.env.MODEL_NAME || 'gpt-4o-mini';

// Health endpoint
app.get('/health', (_req, res) => {
  res.json({ ok: true, model: MODEL_NAME, base: OPENAI_BASE_URL });
});

// Generate endpoint - forwards to provider-compatible chat/completions
app.post('/generate', async (req, res) => {
  try {
    if (!OPENAI_API_KEY) {
      return res.status(400).json({ error: 'Missing OPENAI_API_KEY in environment' });
    }

    const body = req.body || {};
    const messages = body.messages;
    const model = body.model || MODEL_NAME;
    const extra = body.extra || {}; // allow optional pass-through params

    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'Body must include messages: [{ role, content }]' });
    }

    const url = `${OPENAI_BASE_URL}/chat/completions`;

    const providerResp = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENAI_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ model, messages, ...extra })
    });

    const data = await providerResp.json();

    if (!providerResp.ok) {
      // Pass through provider error
      return res.status(providerResp.status).json({ error: data.error || data });
    }

    const text = data?.choices?.[0]?.message?.content || '';
    return res.json({ text });
  } catch (err) {
    return res.status(500).json({ error: 'Upstream error', detail: String(err && err.message || err) });
  }
});

app.listen(PORT, () => {
  console.log(`model-service listening on http://localhost:${PORT}`);
});

