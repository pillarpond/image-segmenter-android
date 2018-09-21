/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package pp.imagesegmenter.tracking;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.util.TypedValue;
import android.widget.Toast;

import pp.imagesegmenter.Deeplab.Recognition;
import pp.imagesegmenter.env.BorderedText;
import pp.imagesegmenter.env.ImageUtils;
import pp.imagesegmenter.env.Logger;

import java.util.LinkedList;
import java.util.List;

/**
 * A tracker wrapping ObjectTracker that also handles non-max suppression and matching existing
 * objects to new detections.
 */
public class MultiBoxTracker {
    private final Logger logger = new Logger();

    private static final float TEXT_SIZE_DIP = 18;

    // Maximum percentage of a box that can be overlapped by another box at detection time. Otherwise
    // the lower scored box (new or old) will be removed.
    private static final float MAX_OVERLAP = 0.2f;

    private static final float MIN_SIZE = 16.0f;

    // Allow replacement of the tracked box with new results if
    // correlation has dropped below this level.
    private static final float MARGINAL_CORRELATION = 0.75f;

    // Consider object to be lost if correlation falls below this threshold.
    private static final float MIN_CORRELATION = 0.3f;

    private static final int MAX_OBJECT = 16;

    private ObjectTracker objectTracker;

    private final List<RectF> screenRects = new LinkedList<RectF>();

    private static class TrackedRecognition {
        ObjectTracker.TrackedObject trackedObject;
        RectF location;
        Bitmap bitmap;
    }

    private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();

    private final BorderedText borderedText;

    private Matrix frameToCanvasMatrix;

    private int frameWidth;
    private int frameHeight;

    private int sensorOrientation;
    private Context context;

    public MultiBoxTracker(final Context context) {
        this.context = context;

        float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
    }

    private Matrix getFrameToCanvasMatrix() {
        return frameToCanvasMatrix;
    }

    public synchronized void drawDebug(final Canvas canvas) {
        final Paint textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(60.0f);

        final Paint boxPaint = new Paint();
        boxPaint.setColor(Color.RED);
        boxPaint.setAlpha(200);
        boxPaint.setStyle(Style.STROKE);

        for (final RectF rect : screenRects) {
            canvas.drawRect(rect, boxPaint);
        }

        if (objectTracker == null) {
            return;
        }

        // Draw correlations.
        for (final TrackedRecognition recognition : trackedObjects) {
            final ObjectTracker.TrackedObject trackedObject = recognition.trackedObject;

            final RectF trackedPos = trackedObject.getTrackedPositionInPreviewFrame();

            if (getFrameToCanvasMatrix().mapRect(trackedPos)) {
                final String labelString = String.format("%.2f", trackedObject.getCurrentCorrelation());
                borderedText.drawText(canvas, trackedPos.right, trackedPos.bottom, labelString);
            }
        }

        final Matrix matrix = getFrameToCanvasMatrix();
        objectTracker.drawDebug(canvas, matrix);
    }

    public synchronized void trackResults(
            final List<Recognition> results, final byte[] frame, final long timestamp) {
        logger.i("Processing %d results from %d", results.size(), timestamp);
        processResults(timestamp, results, frame);
    }

    public synchronized void draw(final Canvas canvas) {
        final boolean rotated = sensorOrientation % 180 == 90;
        final float multiplier =
                Math.min(canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
                        canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
        frameToCanvasMatrix =
                ImageUtils.getTransformationMatrix(
                        frameWidth,
                        frameHeight,
                        (int) (multiplier * (rotated ? frameHeight : frameWidth)),
                        (int) (multiplier * (rotated ? frameWidth : frameHeight)),
                        sensorOrientation,
                        false);
        for (final TrackedRecognition recognition : trackedObjects) {
            final RectF trackedPos =
                    (objectTracker != null)
                            ? recognition.trackedObject.getTrackedPositionInPreviewFrame()
                            : new RectF(recognition.location);

            getFrameToCanvasMatrix().mapRect(trackedPos);

            Bitmap mask = Bitmap.createBitmap((int) trackedPos.width(), (int) trackedPos.height(), Bitmap.Config.ARGB_8888);
            Canvas maskCanvas = new Canvas(mask);
            Matrix maskMatrix =
                    ImageUtils.getTransformationMatrix(
                            (int) recognition.location.width(), (int) recognition.location.height(),
                            (int) trackedPos.width(), (int) trackedPos.height(),
                            sensorOrientation, false);
            maskCanvas.drawBitmap(recognition.bitmap, maskMatrix, null);

            canvas.drawBitmap(mask, null, trackedPos, null);
        }
    }

    private boolean initialized = false;

    public synchronized void onFrame(
            final int w,
            final int h,
            final int rowStride,
            final int sensorOrienation,
            final byte[] frame,
            final long timestamp) {
        if (objectTracker == null && !initialized) {
            ObjectTracker.clearInstance();

            logger.i("Initializing ObjectTracker: %dx%d", w, h);
            objectTracker = ObjectTracker.getInstance(w, h, rowStride, true);
            frameWidth = w;
            frameHeight = h;
            this.sensorOrientation = sensorOrienation;
            initialized = true;

            if (objectTracker == null) {
                String message =
                        "Object tracking support not found. "
                                + "See tensorflow/examples/android/README.md for details.";
                Toast.makeText(context, message, Toast.LENGTH_LONG).show();
                logger.e(message);
            }
        }

        if (objectTracker == null) {
            return;
        }

        objectTracker.nextFrame(frame, null, timestamp, null, true);

        // Clean up any objects not worth tracking any more.
        final LinkedList<TrackedRecognition> copyList =
                new LinkedList<TrackedRecognition>(trackedObjects);
        for (final TrackedRecognition recognition : copyList) {
            final ObjectTracker.TrackedObject trackedObject = recognition.trackedObject;
            final float correlation = trackedObject.getCurrentCorrelation();
            if (correlation < MIN_CORRELATION) {
                logger.v("Removing tracked object %s because NCC is %.2f", trackedObject, correlation);
                trackedObject.stopTracking();
                trackedObjects.remove(recognition);
            }
        }
    }

    private void processResults(
            final long timestamp, final List<Recognition> results, final byte[] originalFrame) {
        final List<Recognition> rectsToTrack = new LinkedList<>();

        screenRects.clear();
        final Matrix rgbFrameToScreen = new Matrix(getFrameToCanvasMatrix());

        for (final Recognition result : results) {
            if (result.getLocation() == null) {
                continue;
            }
            final RectF detectionFrameRect = new RectF(result.getLocation());

            final RectF detectionScreenRect = new RectF();
            rgbFrameToScreen.mapRect(detectionScreenRect, detectionFrameRect);

            logger.v(
                    "Result! Frame: " + result.getLocation() + " mapped to screen:" + detectionScreenRect);

            screenRects.add(detectionScreenRect);

            if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE) {
                logger.w("Degenerate rectangle! " + detectionFrameRect);
                continue;
            }

            rectsToTrack.add(result);
        }

        if (rectsToTrack.isEmpty()) {
            logger.v("Nothing to track, aborting.");
            return;
        }

        if (objectTracker == null) {
            trackedObjects.clear();
            for (final Recognition potential : rectsToTrack) {
                final TrackedRecognition trackedRecognition = new TrackedRecognition();
                trackedRecognition.location = new RectF(potential.getLocation());
                trackedRecognition.trackedObject = null;
                trackedRecognition.bitmap = potential.getBitmap();
                trackedObjects.add(trackedRecognition);

                if (trackedObjects.size() >= MAX_OBJECT) {
                    break;
                }
            }
            return;
        }

        logger.i("%d rects to track", rectsToTrack.size());
        for (final Recognition potential : rectsToTrack) {
            handleDetection(originalFrame, timestamp, potential);
        }
    }

    private void handleDetection(
            final byte[] frameCopy, final long timestamp, final Recognition potential) {
        final ObjectTracker.TrackedObject potentialObject =
                objectTracker.trackObject(potential.getLocation(), timestamp, frameCopy);

        final float potentialCorrelation = potentialObject.getCurrentCorrelation();
        logger.v(
                "Tracked object went from %s to %s with correlation %.2f",
                potential, potentialObject.getTrackedPositionInPreviewFrame(), potentialCorrelation);

        if (potentialCorrelation < MARGINAL_CORRELATION) {
            logger.v("Correlation too low to begin tracking %s.", potentialObject);
            potentialObject.stopTracking();
            return;
        }

        final List<TrackedRecognition> removeList = new LinkedList<TrackedRecognition>();

        float maxIntersect = 0.0f;

        // This is the current tracked object whose color we will take. If left null we'll take the
        // first one from the color queue.
        TrackedRecognition recogToReplace = null;

        // Look for intersections that will be overridden by this object or an intersection that would
        // prevent this one from being placed.
        for (final TrackedRecognition trackedRecognition : trackedObjects) {
            final RectF a = trackedRecognition.trackedObject.getTrackedPositionInPreviewFrame();
            final RectF b = potentialObject.getTrackedPositionInPreviewFrame();
            final RectF intersection = new RectF();
            final boolean intersects = intersection.setIntersect(a, b);

            final float intersectArea = intersection.width() * intersection.height();
            final float totalArea = a.width() * a.height() + b.width() * b.height() - intersectArea;
            final float intersectOverUnion = intersectArea / totalArea;

            // If there is an intersection with this currently tracked box above the maximum overlap
            // percentage allowed, either the new recognition needs to be dismissed or the old
            // recognition needs to be removed and possibly replaced with the new one.
            if (intersects && intersectOverUnion > MAX_OVERLAP) {
                removeList.add(trackedRecognition);

                // Let the previously tracked object with max intersection amount donate its color to
                // the new object.
                if (intersectOverUnion > maxIntersect) {
                    maxIntersect = intersectOverUnion;
                    recogToReplace = trackedRecognition;
                }
            }
        }

        // Remove everything that got intersected.
        for (final TrackedRecognition trackedRecognition : removeList) {
            logger.v(
                    "Removing tracked object %s with detection correlation %.2f",
                    trackedRecognition.trackedObject,
                    trackedRecognition.trackedObject.getCurrentCorrelation());
            trackedRecognition.trackedObject.stopTracking();
            trackedObjects.remove(trackedRecognition);
        }

        // Finally safe to say we can track this object.
        logger.v(
                "Tracking object %s with detection at position %s",
                potentialObject,
                potential.getLocation());
        final TrackedRecognition trackedRecognition = new TrackedRecognition();
        trackedRecognition.location = potential.getLocation();
        trackedRecognition.trackedObject = potentialObject;
        trackedRecognition.bitmap = potential.getBitmap();
        trackedObjects.add(trackedRecognition);
    }
}
