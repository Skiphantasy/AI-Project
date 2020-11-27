const video = document.getElementById('video')
Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri("{{ url_for('static', filename='models/') }}"),
    faceapi.nets.faceRecognitionNet.loadFromUri("{{ url_for('static', filename='models/') }}")
]).then(startVideo)

function startVideo() {
    navigator.getUserMedia(
        {
            video: {}
        },
        stream => video.srcObject = stream,
        err => console.error(err)
    )
}

video.addEventListener('play', () => {
    const canvas = faceapi.createCanvasFromMedia(video)
    document.body.append(canvas)
    const displaySize = { width: video.width, height: video.height }
    faceapi.matchDimensions(canvas, displaySize)
    setInterval(async () => {
        const detections = await faceapi.detectAllFaces(video, new faceapi.SsdMobilenetv1Options({minConfidence: 0.18, maxFaces: 20}))
        if (detections.length > 0) {
            extractFaceFromBox(img, detections[0]._box)
        }
        const resizedDetections = faceapi.resizeResults(detections, displaySize)
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
        faceapi.draw.drawDetections(canvas, resizedDetections)
    }, 100)
})

/* 
    Extrae las caras detectadas por Face-API y las anida en un elemento <div> del DOM
*/
async function extractFaceFromBox(inputImage, box) {
    const regionsToExtract = [
        new faceapi.Rect(box.x, box.y, box.width, box.height)
    ]

    let faceImages = await faceapi.extractFaces(inputImage, regionsToExtract)

    if (faceImages.length == 0) {
        console.log('Face not found')
    }
    else {
        document.getElementById("resultados").innerHTML = "";
        faceImages.forEach(e => {
            document.getElementById("resultados").appendChild(e);
        })
    }
}

/*
    (En esta versión del código no se está utilizando)
    Realiza el preprocesado de una imagen para el modelo entrenado, que al final lo deja con un shape de [1, 96, 96, 3]
*/
function preprocess(img) {
    const example = tf.browser.fromPixels(img);
    example.print();
    const resized = tf.image.resizeBilinear(example, [96, 96]).toFloat()
    const offset = tf.scalar(255.0);
    const normalized = tf.scalar(1.0).sub(resized.div(offset));
    const batched = normalized.expandDims(0)
    return batched
}