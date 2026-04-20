#!/usr/bin/env python

"""
SageMaker inference server for outlier detection using FastAPI.
Implements required /ping and /invocations endpoints.

This service uses a pre-built lookup table to detect price outliers
based on segment-level statistics.
"""
import json
import sys
import traceback
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from deployment.src.schemas import PredictionRequest, PredictionResponse
from deployment.src.model_loader import load_model
from deployment.src.inference import process_single_request
from deployment.src import feedback_loop

app = FastAPI(
    title="Price Realism Outlier Detection",
    description="SageMaker inference server for outlier detection using lookup tables",
    version="1.0.0"
)

artifacts = None


def _load_artifacts_with_feedback():
    global artifacts
    artifacts = load_model()
    feedback_loop._load_feedback_from_s3()
    return artifacts


@app.get('/ping')
async def ping():
    """Health check endpoint."""
    try:
        if artifacts is None:
            _load_artifacts_with_feedback()
        return JSONResponse(content={'status': 'healthy'}, status_code=200)
    except Exception as e:
        print(f"Health check failed: {traceback.format_exc()}", file=sys.stderr)
        return JSONResponse(content={'status': 'unhealthy', 'error': str(e)}, status_code=500)


@app.post('/invocations')
async def invocations(request: Request):
    """
    Main inference endpoint for outlier detection.
    Accepts single request or batch of requests.
    """
    global artifacts
    try:
        if artifacts is None:
            _load_artifacts_with_feedback()

        feedback_loop.maybe_refresh_feedback()

        content_type = request.headers.get('content-type', 'application/json')
        body = await request.body()

        if content_type == 'application/json':
            data = json.loads(body)
        else:
            raise HTTPException(status_code=415, detail="Unsupported Media Type. Use application/json")

        if isinstance(data, dict) and data.get('action') == 'force_refresh_feedback':
            print("[ACTION] Force refresh feedback requested via /invocations", file=sys.stderr)
            feedback_loop._load_feedback_from_s3()
            print(f"[DEBUG] After refresh - feedback_lookup has {len(feedback_loop.feedback_lookup['by_listing_id'])} listings, {len(feedback_loop.feedback_lookup['by_segment_key'])} segments", file=sys.stderr)
            return {
                "action": "force_refresh_feedback",
                "status": "success",
                "feedback_listings": len(feedback_loop.feedback_lookup['by_listing_id']),
                "feedback_segments": len(feedback_loop.feedback_lookup['by_segment_key']),
                "timestamp": datetime.now().isoformat()
            }

        lookup_dict = artifacts["lookup_table"].set_index('segment_key').to_dict('index')

        is_batch = isinstance(data, list)
        requests = data if is_batch else [data]

        results = []
        for req_data in requests:
            try:
                validated_request = PredictionRequest(**req_data)
                result = process_single_request(validated_request.dict(), lookup_dict, artifacts, feedback_loop.feedback_lookup)
                results.append(result)

            except ValueError as ve:
                error_response = {"error": str(ve)}
                results.append(error_response)
            except Exception as e:
                error_response = {"error": f"Processing error: {str(e)}"}
                results.append(error_response)

        response_data = results if is_batch else results[0]
        return JSONResponse(content=response_data)

    except json.JSONDecodeError as e:
        return JSONResponse(
            content={'error': f'Invalid JSON: {str(e)}'},
            status_code=400
        )
    except Exception as e:
        print(f"Error: {traceback.format_exc()}", file=sys.stderr)
        return JSONResponse(content={'error': str(e)}, status_code=500)


@app.post('/refresh_feedback')
async def refresh_feedback():
    """
    Force-refresh the CX feedback lookup table from S3.
    This endpoint triggers a synchronous reload.
    """
    try:
        feedback_loop._load_feedback_from_s3()
        return JSONResponse(content={
            'status': 'refreshed',
            'feedback_listings': len(feedback_loop.feedback_lookup['by_listing_id']),
            'feedback_segments': len(feedback_loop.feedback_lookup['by_segment_key']),
            'last_update_ts': feedback_loop.feedback_last_update_ts.isoformat() if feedback_loop.feedback_last_update_ts else None,
        }, status_code=200)
    except Exception as e:
        print(f"[FEEDBACK] Force refresh failed: {traceback.format_exc()}", file=sys.stderr)
        return JSONResponse(content={'error': str(e)}, status_code=500)


@app.get('/feedback_status')
async def feedback_status():
    """Return current feedback lookup table status."""
    return JSONResponse(content={
        'feedback_listings': len(feedback_loop.feedback_lookup['by_listing_id']),
        'feedback_segments': len(feedback_loop.feedback_lookup['by_segment_key']),
        'last_update_ts': feedback_loop.feedback_last_update_ts.isoformat() if feedback_loop.feedback_last_update_ts else None,
        'is_updating': feedback_loop._feedback_updating,
    }, status_code=200)


if __name__ == "__main__":
    import uvicorn

    print("Starting server for local testing...")
    print("Note: In SageMaker, this will be started by the container")
    uvicorn.run(app, host="0.0.0.0", port=8080)
