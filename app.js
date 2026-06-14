const CLASS_NAMES = ['happy', 'angry', 'alert', 'relax'];

let model;
const videoElement = document.getElementById('webcam');
const startBtn = document.getElementById('start-btn');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot');

// Async initialization sequence
async function loadModel() {
    try {
        // Point to the local converted folder
        model = await tf.loadLayersModel('https://bhavyapatel06.github.io/Real-Time-Dog-Behaviour-Information/model.json');
        
        statusText.innerText = "System ready. Initialize scanner window.";
        statusDot.className = "status-indicator ready";
        startBtn.disabled = false;
    } catch (error) {
        statusText.innerText = "Error loading intelligence package framework.";
        statusDot.className = "status-indicator error";
        console.error(error);
    }
}

// Adaptive multi-device camera configuration initialization
async function setupCamera() {
    startBtn.style.display = 'none';
    videoElement.style.display = 'block';
    statusText.innerText = "Configuring media capture hardware streams...";

    // Strategy 1: Attempt mobile optimal landscape rear environment camera configuration
    let constraints = {
        video: { facingMode: 'environment', width: 224, height: 224 },
        audio: false
    };

    try {
        await startMediaStream(constraints);
    } catch (mobileError) {
        console.warn("Rear environment camera not found. Executing standard fallback protocol...");
        
        // Strategy 2: Fallback to whatever camera exists (e.g. standard laptop webcam configuration)
        constraints = {
            video: true,
            audio: false
        };
        
        try {
            await startMediaStream(constraints);
        } catch (laptopError) {
            statusText.innerText = "Capture initialization failure: Hardware unavailable.";
            statusDot.className = "status-indicator error";
            console.error(laptopError);
        }
    }
}

// Secondary execution engine for video bindings
async function startMediaStream(config) {
    const stream = await navigator.mediaDevices.getUserMedia(config);
    videoElement.srcObject = stream;
    
    videoElement.onloadedmetadata = () => {
        statusText.innerText = "Live processing pipeline telemetry active";
        statusDot.className = "status-indicator active";
        predictFrame();
    };
}

// Processing cycle looping framework
async function predictFrame() {
    if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
        const predictions = tf.tidy(() => {
            const img = tf.browser.fromPixels(videoElement);
            const resized = tf.image.resizeBilinear(img, [224, 224]);
            const normalized = resized.div(tf.scalar(127.5)).sub(tf.scalar(1));
            const batched = normalized.expandDims(0);
            
            return model.predict(batched).dataSync();
        });

        updateDashboard(predictions);
    }
    requestAnimationFrame(predictFrame);
}

// Interface display updater 
function updateDashboard(scores) {
    const targetEmotions = {
        happy: scores[0],
        angry: scores[1],
        alert: scores[2],
        relax: scores[3]
    };

    // Determine the current absolute maximum value for active state styling updates
    let maxScore = -1;
    let winningEmotion = "";
    
    CLASS_NAMES.forEach((emotion, index) => {
        const currentScore = scores[index];
        if (currentScore > maxScore) {
            maxScore = currentScore;
            winningEmotion = emotion;
        }
    });

    // Update each visual metric row block completely
    CLASS_NAMES.forEach((emotion) => {
        const pctValue = (targetEmotions[emotion] * 100).toFixed(1);
        
        // Target specific nodes
        const barNode = document.getElementById(`bar-${emotion}`);
        const textNode = document.getElementById(`pct-happy`);
        const cardNode = document.getElementById(`card-${emotion}`);
        
        // Correct individual target mapping updates
        document.getElementById(`bar-${emotion}`).style.width = `${pctValue}%`;
        document.getElementById(`pct-${emotion}`).innerText = `${pctValue}%`;
        
        // Highlight active matrix card element systematically
        if (emotion === winningEmotion) {
            document.getElementById(`card-${emotion}`).classList.add('active-state');
        } else {
            document.getElementById(`card-${emotion}`).classList.remove('active-state');
        }
    });
}

startBtn.addEventListener('click', setupCamera);
loadModel();
