import asyncio
import websockets
import json
import random


MAX_COORDINATES = 5

async def send_coordinates():
    """Envía coordenadas aleatorias al WebSocket cada 2 segundos."""
    async with websockets.connect("ws://localhost:8765") as websocket:
        num_coordinates = 0
        while num_coordinates < MAX_COORDINATES:
            coordinates = {
                "x": round(random.uniform(-5, 5), 2),
                "y": 0.9,
                "z": round(random.uniform(-5, 5), 2)
            }
            await websocket.send(json.dumps(coordinates))
            print(f"Enviando: {coordinates}")
            num_coordinates += 1
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(send_coordinates())