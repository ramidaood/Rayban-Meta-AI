import asyncio
import websockets
import json
import os
import socket

DIRECTION_FILE = "last_direction.txt"
WS_PORT = 4040

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception:
        return "127.0.0.1"  # Fallback to localhost

WS_HOST = "0.0.0.0"  # Bind to all interfaces

async def direction_broadcaster(websocket):
    print("Client connected")
    last_sent = None
    try:
        while True:
            # Read the latest direction from file
            if os.path.exists(DIRECTION_FILE):
                with open(DIRECTION_FILE, "r") as f:
                    direction = f.read().strip()
            else:
                direction = "None"
            # Only send if changed
            if direction != last_sent:
                await websocket.send(json.dumps({"direction": direction}))
                print(f"Sent direction: {direction}")
                last_sent = direction
            await asyncio.sleep(0.1)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    local_ip = get_local_ip()
    print(f"WebSocket server running on:")
    print(f"  - ws://{local_ip}:{WS_PORT} (external access)")
    print(f"  - ws://127.0.0.1:{WS_PORT} (localhost)")
    print(f"  - ws://0.0.0.0:{WS_PORT} (all interfaces)")
    
    try:
        async with websockets.serve(direction_broadcaster, WS_HOST, WS_PORT):
            await asyncio.Future()  # run forever
    except OSError as e:
        print(f"Error starting server: {e}")
        print("Trying with localhost instead...")
        async with websockets.serve(direction_broadcaster, "127.0.0.1", WS_PORT):
            print(f"WebSocket server running on ws://127.0.0.1:{WS_PORT}")
            await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main()) 