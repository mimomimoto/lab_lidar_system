import asyncio
import base64
from fastapi import APIRouter, Body, FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
from typing import Any, Dict, List, Union
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import open3d as o3d
import matplotlib.pyplot as plt

# frame_pcd = o3d.io.read_point_cloud("/ws/lidar_server/base/frame_lab.pcd")
# frame_pcd = frame_pcd.voxel_down_sample(voxel_size=0.2)
# frame_pcd.paint_uniform_color([1, 1, 1])

# frame_pcd_numpy = np.asarray(frame_pcd.points, dtype='float32')
# pcd_color = np.asarray(frame_pcd.colors, dtype='float32')
# frame_colors_np = np.asarray(pcd_color * 255, dtype='uint8')



templates = Jinja2Templates(directory="templates")
stored_data: Dict[str, Any] = {"pointcloud": {}, "log": "", "clearLog": False}
class PointCloudData(BaseModel):
    name: str
    points: str
    colors: str


class PointCloudDataArray(BaseModel):
    array: List[PointCloudData]
    clear: bool

def _encode(s: bytes) -> str:
    return base64.b64encode(s).decode("utf-8")


def _decode(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))
app = FastAPI()
app.mount(
    '/templates/static', 
    StaticFiles(directory="templates/static"), 
    name='static'
    )

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request
    })


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            data_pack: List[List[np.ndarray]] = [[], []]
            data_pack[0] = stored_data["pointcloud"][:][0]
            data_pack[1] = stored_data["color_data"][:][0]
            

            
            send_data: List[Union[List, str]] = [[], []]
        
            
            if len(data_pack[0]) > 0:
                send_data[0] = _encode(np.concatenate(data_pack[0], axis=0).tobytes("C"))
            if len(data_pack[1]) > 0:
                send_data[1] = _encode(np.concatenate(data_pack[1], axis=0).tobytes("C"))
            await ws.send_json(send_data)
    except:
        pass
        # await ws.close()
        
@app.post("/pointcloud/store")
async def store_data(data: PointCloudData):
    points = np.frombuffer(_decode(data.points), dtype=np.float32)
    points = points.reshape((-1, 3))
    colors = np.frombuffer(_decode(data.colors), dtype=np.uint8)
    colors = colors.reshape((-1, 3))
    all_points = points
    all_color = colors
    # all_points = np.concatenate([points, frame_pcd_numpy])
    # all_color = np.concatenate([colors, frame_colors_np])
    stored_data["pointcloud"] = [all_points]
    stored_data["color_data"] = [all_color]
    # print(stored_data)
    return {"res": "ok", "name": data.name}