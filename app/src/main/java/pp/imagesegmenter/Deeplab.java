/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

package pp.imagesegmenter;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.RectF;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

import pp.imagesegmenter.env.ImageUtils;

public class Deeplab {
    /**
     * An immutable result returned by a Deeplap describing what was recognized.
     */
    public class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        private Bitmap bitmap;

        Recognition(
                final String id, final RectF location, final Bitmap bitmap) {
            this.id = id;
            this.location = location;
            this.bitmap = bitmap;
        }

        public String getId() {
            return id;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public Bitmap getBitmap() {return bitmap;}

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }

    private static final String MODEL_FILE = "file:///android_asset/frozen_inference_graph.pb";
    private static final String INPUT_NODE = "ImageTensor";
    private static final String OUTPUT_NODE = "SemanticPredictions";
    private static final int[] colormap = {
            0x00000000,     //background
            0x99ffe119,     //aeroplane
            0x993cb44b,     //bicycle
            0x99808000,     //bird
            0x99008080,     //boat
            0x99000080,     //bottle
            0x99e6194b,     //bus
            0x99f58230,     //car
            0x99800000,     //cat
            0x99d2f53c,     //chair
            0x99aa6e28,     //cow
            0x9946f0f0,     //diningtable
            0x99911eb4,     //dog
            0x99f032e6,     //horse
            0x990082c8,     //motobike
            0x99fabebe,     //person
            0x99ffd7b4,     //pottedplant
            0x99808080,     //sheep
            0x99fffac8,     //sofa
            0x99aaffc3,     //train
            0x99e6beff      //tv
    };

    private int sensorOrientation;
    private int width;
    private int height;

    private int[] intValues;
    private byte[] byteValues;
    private int[] outputValues;
    private TensorFlowInferenceInterface inferenceInterface;

    private Stack<Point> pointStack;
    private Stack<Point> maskStack;

    Deeplab(AssetManager assetManager,
            int inputWidth,
            int inputHeight,
            int sensorOrientation) {
        inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);

        final Graph g = inferenceInterface.graph();

        // The inputName node has a shape of [N, H, W, C], where
        // N is the batch size
        // H = W are the height and width
        // C is the number of channels (3 for our purposes - RGB)
        final Operation
                inputOp1 = g.operation(INPUT_NODE);
        if (inputOp1 == null) {
            throw new RuntimeException("Failed to find input Node '" + INPUT_NODE + "'");
        }

        final Operation outputOp1 = g.operation(OUTPUT_NODE);
        if (outputOp1 == null) {
            throw new RuntimeException("Failed to find output Node'" + OUTPUT_NODE + "'");
        }

        this.sensorOrientation = sensorOrientation;
        width = inputWidth;
        height = inputHeight;

        // Pre-allocate buffers.
        intValues = new int[inputWidth * inputHeight];
        byteValues = new byte[inputWidth * inputHeight * 3];
        outputValues = new int[inputWidth * inputHeight];

        pointStack = new Stack<>();
        maskStack = new Stack<>();
    }

    List<Recognition> segment(Bitmap bitmap, Matrix matrix) {
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            byteValues[i * 3] = (byte) ((val >> 16) & 0xFF);
            byteValues[i * 3 + 1] = (byte) ((val >> 8) & 0xFF);
            byteValues[i * 3 + 2] = (byte) (val & 0xFF);
        }

        // Copy the input data into TensorFlow.
        inferenceInterface.feed(
                INPUT_NODE, byteValues, 1, bitmap.getHeight(), bitmap.getWidth(), 3);

        inferenceInterface.run(new String[]{OUTPUT_NODE});
        inferenceInterface.fetch(OUTPUT_NODE, outputValues);

        final List<Recognition> mappedRecognitions = new LinkedList<>();

        int cnt = 0;
        for (int col = 0; col < height; col++) {
            for (int row = 0; row < width; row++) {
                if (outputValues[col * width + row] == 0) continue;

                RectF rectF = new RectF();
                rectF.top = col;
                rectF.bottom = col + 1;
                rectF.left = row;
                rectF.right = row + 1;

                int id = outputValues[col * width + row];

                floodFill(row, col, id, rectF);

                Bitmap maskBitmap = createMask(id, matrix, rectF);

                Recognition result =
                        new Recognition("" + cnt++, rectF, maskBitmap);
                mappedRecognitions.add(result);
            }
        }

        return mappedRecognitions;

    }

    private Bitmap createMask(int id, Matrix matrix, RectF rectF) {
        int w = (int) rectF.width();
        int h = (int) rectF.height();

        int top = (int) rectF.top;
        int left = (int) rectF.left;

        Bitmap segBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        int tmpValues[] = new int[w * h];

        while (!maskStack.empty()) {
            Point point = maskStack.pop();
            tmpValues[(point.y - top) * w + point.x - left] = colormap[id];
        }

        segBitmap.setPixels(tmpValues, 0, w, 0, 0, w, h);
        matrix.mapRect(rectF);

        Bitmap mask = Bitmap.createBitmap((int) rectF.width(), (int) rectF.height(), Bitmap.Config.ARGB_8888);

        Matrix maskMatrix = new Matrix();
        ImageUtils.getTransformationMatrix(
                (int) rectF.width(), (int) rectF.height(),
                w, h,
                sensorOrientation, false).invert(maskMatrix);

        Canvas canvas = new Canvas(mask);
        canvas.drawBitmap(segBitmap, maskMatrix, null);

        return mask;
    }

    private void floodFill(int initX, int initY, int val, RectF rectF) {
        outputValues[initY * width + initX] = 0;
        pointStack.push(new Point(initX, initY));

        while (!pointStack.empty()) {
            Point point = pointStack.pop();
            maskStack.push(point);

            int row = point.x;
            int col = point.y;

            if (rectF.top > col) rectF.top = col;
            if (rectF.bottom < col + 1) rectF.bottom = col + 1;
            if (rectF.left > row) rectF.left = row;
            if (rectF.right < row + 1) rectF.right = row + 1;

            if (row > 0 && val == outputValues[col * width + row - 1]) {
                outputValues[col * width + row - 1] = 0;
                pointStack.push(new Point(row - 1, col));
            }
            if (row < width - 1 && val == outputValues[col * width + row + 1]) {
                outputValues[col * width + row + 1] = 0;
                pointStack.push(new Point(row + 1, col));
            }
            if (col > 0 && val == outputValues[(col - 1) * width + row]) {
                outputValues[(col - 1) * width + row] = 0;
                pointStack.push(new Point(row, col - 1));
            }
            if (col < height - 1 && val == outputValues[(col + 1) * width + row]) {
                outputValues[(col + 1) * width + row] = 0;
                pointStack.push(new Point(row, col + 1));
            }
        }
    }

    String getStatString() {
        return inferenceInterface.getStatString();
    }
}
