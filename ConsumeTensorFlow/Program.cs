using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace ConsumeTensorFlow
{
    class Program
    {
        class PredictedImageData
        {
            [ColumnName("box")]
            [VectorType(4)]
            public float[] BoundingBox { get; set; }

            [ColumnName("landmarks")]
            [VectorType(68)]
            public float[] Landmarks { get; set; }

            //[VectorType(1)]
            [ColumnName("prob")]
            public float Probability { get; set; }
        }

        private struct ImageNetSettings
        {
            public const int imageHeight = 224;
            public const int imageWidth = 224;
        }

        public class ImageNetData
        {
            [ImageType(ImageNetSettings.imageHeight, ImageNetSettings.imageWidth)]
            [ColumnName("input")]
            public Bitmap Input { get; set; }

            [ColumnName("image_path")]
            public string ImagePath { get; set; }

            [VectorType(2)]
            [ColumnName("min_size")]
            public float[] MinSize { get; set; }

            [VectorType(2)]
            [ColumnName("factor")]
            public float[] Factor { get; set; }

            [VectorType(4)]
            [ColumnName("thresholds")]
            public float[] Thresholds { get; set; }
        }

        static void Main(string[] args)
        {
            NewMethod();
        }

        private static void NewMethod()
        {
            var mlContext = new MLContext(seed: 1);
            var modelLocation = "mtcnn.pb";
            var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());
            
            var schema = data.Schema;

            var pipeline = mlContext.Transforms.LoadImages(
                    outputColumnName: "input",
                    imageFolder: "C:\\Data",
                    inputColumnName: "image_path")
                .Append(mlContext.Transforms.ResizeImages(
                    outputColumnName: "input",
                    imageWidth: ImageNetSettings.imageWidth,
                    imageHeight: ImageNetSettings.imageHeight,
                    inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(
                    outputColumnName: "input"))
                .Append(mlContext.Model.LoadTensorFlowModel(modelLocation).ScoreTensorFlowModel(
                    outputColumnNames: new[] { "box", "prob", "landmarks" },
                    inputColumnNames: new[] { "input", "min_size", "factor", "thresholds" },
                    addBatchDimensionInput: true));

            var pipelinePreview = pipeline.Preview(data);

            ITransformer model = pipeline.Fit(data);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageNetData, PredictedImageData>(model);

            var schema1 = model.GetOutputSchema(schema);

            var prediction = predictionEngine.Predict(new ImageNetData
            {
                ImagePath = "C:\\Data\\anastasia3.jpg",
                //Input = (Bitmap)Image.FromFile("C:\\Data\\anastasia3.jpg"),
                MinSize = new float[] { 1F, 40F },
                Factor = new float[] { 1F, 0.709F },
                Thresholds = new float[] { 1F, 0.6F, 0.7F, 0.7F }
            });
        }
    }
}
