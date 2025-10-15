"""
Twilio ConversationRelay + Groq LLM - REPLIT VERSION
Real-time AI Voice Assistant

Optimized for Replit deployment with automatic HTTPS URL
"""

import os
import json
import asyncio
import base64
from typing import Dict, List
from fastapi import FastAPI, WebSocket, Request, Form
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv
import httpx
import uvicorn

load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
CARTESIA_API_KEY = os.getenv('CARTESIA_API_KEY')  # Optional
USE_CARTESIA_TTS = os.getenv('USE_CARTESIA_TTS', 'false').lower() == 'true'

# Replit configuration
PORT = int(os.getenv('PORT', 8080))  # Replit uses port 8080
HOST = os.getenv('REPL_SLUG', 'localhost')  # Replit provides this

# AI Configuration
SYSTEM_PROMPT = """You are a helpful, friendly voice assistant.
Keep your responses concise and conversational (2-3 sentences max).
Speak naturally as if having a phone conversation.
Be warm, professional, and helpful."""

GROQ_MODEL = "llama-3.1-70b-versatile"
GROQ_TEMPERATURE = 0.7
GROQ_MAX_TOKENS = 150

# Voice Configuration
DEFAULT_VOICE = "Polly.Matthew"
DEFAULT_LANGUAGE = "en-US"

app = FastAPI()

# Store conversation history per call
conversations: Dict[str, List[Dict]] = {}
call_metadata: Dict[str, Dict] = {}


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Twilio ConversationRelay + Groq AI",
        "platform": "Replit",
        "tts_provider": "Cartesia" if USE_CARTESIA_TTS else "Twilio",
        "supports": ["inbound_calls", "outbound_calls_via_n8n"]
    }


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(
    request: Request,
    CallSid: str = Form(None),
    From: str = Form(None),
    To: str = Form(None),
    Direction: str = Form(None),
    CallStatus: str = Form(None)
):
    """Handle incoming call and return TwiML to connect to ConversationRelay"""
    
    # Log call information
    print("\n" + "="*60)
    print("📞 NEW CALL RECEIVED")
    print("="*60)
    print(f"Call SID: {CallSid}")
    print(f"From: {From}")
    print(f"To: {To}")
    print(f"Direction: {Direction}")
    print(f"Status: {CallStatus}")
    print("="*60 + "\n")
    
    # Store call metadata
    if CallSid:
        call_metadata[CallSid] = {
            "from": From,
            "to": To,
            "direction": Direction,
            "status": CallStatus,
            "greeting_sent": False
        }
    
    # Generate TwiML response
    response = VoiceResponse()
    
    # Get the host for WebSocket URL
    # Replit provides REPL_SLUG and REPL_OWNER for URL construction
    repl_slug = os.getenv('REPL_SLUG')
    repl_owner = os.getenv('REPL_OWNER')
    
    if repl_slug and repl_owner:
        # Replit URL format: https://{slug}.{owner}.repl.co
        host = f"{repl_slug}.{repl_owner}.repl.co"
    else:
        # Fallback to request hostname
        host = request.url.hostname
    
    protocol = "wss"  # Always use secure WebSocket on Replit
    ws_url = f'{protocol}://{host}/conversation-relay'
    
    print(f"🔧 Building TwiML...")
    print(f"   Replit Slug: {repl_slug}")
    print(f"   Replit Owner: {repl_owner}")
    print(f"   Hostname: {host}")
    print(f"   WebSocket URL: {ws_url}")
    
    # Connect to ConversationRelay WebSocket
    connect = Connect()
    connect.conversation_relay(
        url=ws_url,
        voice=DEFAULT_VOICE,
        language=DEFAULT_LANGUAGE,
        dtmf_detection=True,
    )
    response.append(connect)
    
    # Log the TwiML being returned
    twiml_str = str(response)
    print(f"\n📤 Returning TwiML to Twilio:")
    print(f"{twiml_str}")
    print("="*60 + "\n")
    
    return Response(content=twiml_str, media_type="application/xml")


@app.post("/status-callback")
async def handle_status_callback(
    request: Request,
    CallSid: str = Form(None),
    CallStatus: str = Form(None),
    Direction: str = Form(None),
    From: str = Form(None),
    To: str = Form(None),
    Duration: str = Form(None),
    CallDuration: str = Form(None)
):
    """Handle Twilio status callbacks"""
    
    print("\n" + "="*60)
    print(f"📊 CALL STATUS UPDATE: {CallStatus}")
    print("="*60)
    print(f"Call SID: {CallSid}")
    print(f"Direction: {Direction}")
    print(f"From: {From} → To: {To}")
    if Duration:
        print(f"Duration: {Duration} seconds")
    print("="*60 + "\n")
    
    # Update call metadata
    if CallSid and CallSid in call_metadata:
        call_metadata[CallSid]["status"] = CallStatus
        if Duration:
            call_metadata[CallSid]["duration"] = Duration
    
    # Cleanup when call ends
    if CallStatus == "completed":
        print(f"🧹 Call completed - cleaning up data for {CallSid}")
        if CallSid in conversations:
            del conversations[CallSid]
        if CallSid in call_metadata:
            del call_metadata[CallSid]
    
    return {"status": "received"}


@app.websocket("/conversation-relay")
async def conversation_relay_handler(websocket: WebSocket):
    """Handle ConversationRelay WebSocket connection"""
    
    print("\n" + "="*60)
    print("🔌 WEBSOCKET CONNECTION ATTEMPT RECEIVED!")
    print("="*60)
    print(f"Client: {websocket.client}")
    print(f"Headers: {websocket.headers}")
    print("="*60 + "\n")
    
    print("📞 ConversationRelay connection initiated - ACCEPTING...")
    await websocket.accept()
    print("✅ WebSocket ACCEPTED successfully!")
    print("⏳ Waiting for messages from Twilio...")
    
    call_sid = None
    stream_sid = None
    
    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            event_type = data.get('type')
            
            print(f"📩 Received: {event_type}")
            
            if event_type == 'setup':
                call_sid = data.get('callSid')
                print(f"✅ Setup complete for call: {call_sid}")
                
                conversations[call_sid] = []
                
                if not USE_CARTESIA_TTS:
                    config_message = {
                        "type": "config",
                        "voiceConfig": {
                            "voice": DEFAULT_VOICE,
                            "language": DEFAULT_LANGUAGE,
                            "speechRate": "100%",
                            "speechPitch": "+0%"
                        }
                    }
                    await websocket.send_json(config_message)
                    print("🔧 Sent voice configuration")
                
                greeting = get_greeting_message(call_sid)
                await send_response_to_caller(websocket, greeting, call_sid)
                
                if call_sid in call_metadata:
                    call_metadata[call_sid]["greeting_sent"] = True
            
            elif event_type == 'prompt':
                user_text = data.get('voicePrompt', '').strip()
                call_sid = data.get('callSid')
                stream_sid = data.get('streamSid')
                
                if not user_text:
                    print("⚠️  Empty prompt received, skipping")
                    continue
                
                print(f"🗣️  User said: '{user_text}'")
                
                conversations[call_sid].append({
                    "role": "user",
                    "content": user_text
                })
                
                ai_response = await get_groq_response(call_sid)
                print(f"🤖 AI response: '{ai_response}'")
                
                conversations[call_sid].append({
                    "role": "assistant",
                    "content": ai_response
                })
                
                await send_response_to_caller(websocket, ai_response, call_sid, stream_sid)
            
            elif event_type == 'interrupt':
                print("🛑 User interrupted - stopping AI speech")
                stream_sid = data.get('streamSid')
                clear_message = {"type": "clear"}
                await websocket.send_json(clear_message)
            
            elif event_type == 'dtmf':
                digit = data.get('digit')
                print(f"🔢 DTMF received: {digit}")
            
            else:
                print(f"❓ Unknown event type: {event_type}")
    
    except Exception as e:
        print(f"❌ Error in conversation relay: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if call_sid and call_sid in conversations:
            print(f"🧹 Cleaning up conversation for {call_sid}")
            del conversations[call_sid]
        
        print("📞 ConversationRelay connection closed")


def get_greeting_message(call_sid: str) -> str:
    """Generate appropriate greeting based on call direction"""
    metadata = call_metadata.get(call_sid, {})
    direction = metadata.get('direction', 'unknown')
    
    if direction == 'outbound-api':
        print(f"🎯 Outbound call detected - using outbound greeting")
        return ("Hello! This is an automated call from our AI assistant. "
                "I'm here to help you. How can I assist you today?")
    else:
        print(f"📞 Inbound call detected - using inbound greeting")
        return "Hello! I'm your AI assistant. How can I help you today?"


async def get_groq_response(call_sid: str) -> str:
    """Get AI response from Groq LLM"""
    conversation = conversations.get(call_sid, [])
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation
    ]
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": messages,
                    "temperature": GROQ_TEMPERATURE,
                    "max_tokens": GROQ_MAX_TOKENS,
                    "top_p": 1,
                    "stream": False
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            ai_text = result["choices"][0]["message"]["content"]
            
            usage = result.get("usage", {})
            print(f"📊 Groq usage - Prompt: {usage.get('prompt_tokens')}, "
                  f"Completion: {usage.get('completion_tokens')}, "
                  f"Total: {usage.get('total_tokens')}")
            
            return ai_text
    
    except httpx.TimeoutException:
        print("⏱️  Groq API timeout")
        return "I'm sorry, I'm having trouble processing that right now. Could you try again?"
    
    except httpx.HTTPStatusError as e:
        print(f"❌ Groq API error: {e.response.status_code} - {e.response.text}")
        return "I apologize, I'm experiencing technical difficulties. Please try again in a moment."
    
    except Exception as e:
        print(f"❌ Unexpected error calling Groq: {e}")
        return "I'm sorry, something went wrong. Could you repeat that?"


async def send_response_to_caller(
    websocket: WebSocket,
    text: str,
    call_sid: str,
    stream_sid: str = None
):
    """Send AI response to caller"""
    await send_twilio_tts(websocket, text)


async def send_twilio_tts(websocket: WebSocket, text: str):
    """Send text to Twilio for TTS conversion and playback"""
    prompt_message = {
        "type": "prompt",
        "voicePrompt": text
    }
    
    await websocket.send_json(prompt_message)
    print(f"📤 Sent to Twilio TTS: '{text}'")


if __name__ == "__main__":
    # Validate configuration
    if not GROQ_API_KEY:
        raise ValueError("❌ GROQ_API_KEY not found in environment variables")
    
    print("=" * 60)
    print("🚀 Starting Twilio ConversationRelay AI Voice Assistant")
    print("=" * 60)
    print(f"🌐 Platform: Replit")
    print(f"📊 LLM Provider: Groq ({GROQ_MODEL})")
    print(f"🔊 TTS Provider: {'Cartesia' if USE_CARTESIA_TTS else 'Twilio'}")
    print(f"🎙️  Voice: {DEFAULT_VOICE}")
    print(f"🔌 Port: {PORT}")
    print(f"🌐 Endpoints:")
    print(f"   - Health: /")
    print(f"   - Webhook: /incoming-call")
    print(f"   - Status Callback: /status-callback")
    print(f"   - WebSocket: /conversation-relay")
    print("=" * 60)
    print("✅ Supports BOTH inbound and outbound calls!")
    print("=" * 60)
    
    # Run with Replit-specific configuration
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )
