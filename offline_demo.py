import json
from src.utils.zone_events import feet_in_zone

def run():
    zone=[20,20,80,80]
    boxes=[[0,0,10,10],[25,25,35,35],[60,60,70,70],[90,90,100,100]]
    states=[feet_in_zone(box,zone,0.5) for box in boxes]
    return {"mode":"synthetic-feet-box-replay","frame_zone_states":states,
            "detector_inference_executed":False,"weights_required_for_video_pipeline":True}

if __name__=="__main__": print(json.dumps(run(),indent=2))
