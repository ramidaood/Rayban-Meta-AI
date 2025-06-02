import asyncio
import websockets
import json
import os

DIRECTION_FILE = "last_direction.txt"
WS_HOST = "0.0.0.0"
WS_PORT = 4000

async def direction_broadcaster(websocket, path):
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
    print(f"WebSocket server running on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(direction_broadcaster, WS_HOST, WS_PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main()) 