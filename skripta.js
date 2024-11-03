const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0;
let lastY = 0;

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = [e.pageX - canvas.offsetLeft, e.pageY - canvas.offsetTop];
}

function draw(e) {
    if (!isDrawing) return;

    const x = e.pageX - canvas.offsetLeft;
    const y = e.pageY - canvas.offsetTop;
    const color = '#000000';

    ctx.lineWidth = 13;
    ctx.lineCap = 'round';
    ctx.strokeStyle = color;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();

    [lastX, lastY] = [x, y];
}

function stopDrawing() {
    isDrawing = false;
    [lastX, lastY] = [0, 0];
}

document.getElementById('saveButton').addEventListener('click', saveImage);
document.getElementById('runPythonButton').addEventListener('click', train);
document.getElementById('resetButton').addEventListener('click', resetCanvas); // Dodat event listener


function train() {
    fetch('server5.php', {
        method: 'POST'
    })
    .then(response => response.text())
    .then(message => {
        // Prikazi poruku unutar div-a sa ID-om "answer"
        document.getElementById('answer').textContent = message;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function saveImage() {
    console.log("SS")
    // Save the current canvas as an image with white background
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.fillStyle = '#FFFFFF'; // White background color
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
    tempCtx.drawImage(canvas, 0, 0);

    const image = tempCanvas.toDataURL('image/png');
    const formData = new FormData();
    formData.append('imageData', image);

    fetch('server2.php', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(message => {
        alert(message);
    })
    .catch(error => {
        console.error('Error:', error);
    });

    fetch('server4.php', {
        method: 'POST'
    })
    .then(response => response.text())
    .then(message => {
        // Prikazi poruku unutar div-a sa ID-om "answer"
        document.getElementById('answer').textContent = message;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function resetCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

