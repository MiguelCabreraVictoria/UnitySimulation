"""
Miguel Angel Cabrera Victoria
A01782982

Script: Servidor WebSockets
    
"""


import asyncio
import websockets
import json 


connected_clients = set()


async def ws_server(websocket):
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            print(f"Mensaje recibido: {message}")
            # Enviar el mensaje a todos los clientes conectados
            for client in connected_clients:
                if client != websocket:
                    await client.send(message)
    except websockets.exceptions.ConnectionClosed:
        print("Cliente desconectado.")
    finally:
        connected_clients.remove(websocket)


async def main():
    server = await websockets.serve(ws_server, "localhost", 8765)
    print("Servidor WebSocket corriendo en ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
