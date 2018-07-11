import json

from threading import Thread
from websocket_server import WebsocketServer


class Publisher(Thread):
    def __init__(self, host, port):
        super().__init__(daemon=True)
        self._server = WebsocketServer(port, host)
        self.start()

    def run(self):
        self._server.run_forever()

    def send(self, message):
        self._server.send_message_to_all(json.dumps(message))


def poses_to_dto(poses):
    return {
        pose_id: {
            part_id: {
                'x': part.x,
                'y': part.y,
                'score': part.score
            } for part_id, part in pose.body_parts.items()
        } for pose_id, pose in enumerate(poses)
    }
