const state = {
    timers: {},
    assets: null,
};

const $ = (id) => document.getElementById(id);

function fillSelect(select, items, emptyText) {
    select.innerHTML = "";

    if (!items.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = emptyText;
        option.disabled = true;
        option.selected = true;
        select.appendChild(option);
        select.disabled = true;
        return;
    }

    select.disabled = false;
    items.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.name;
        option.textContent = item.name;
        if (item.url) option.dataset.url = item.url;
        select.appendChild(option);
    });
}

function selectIfExists(select, value) {
    if (!value) return;
    const option = [...select.options].find((item) => item.value === value);
    if (option) select.value = value;
}

function updatePreview(select, image) {
    const option = select.selectedOptions[0];
    if (!option || !option.dataset.url) {
        image.removeAttribute("src");
        image.classList.remove("ready");
        return;
    }
    image.src = `${option.dataset.url}?t=${Date.now()}`;
    image.classList.add("ready");
}

function updateRunButtons() {
    const hasInputs = state.assets && state.assets.inputs.length > 0;
    const hasModels = state.assets && state.assets.fast.models.length > 0;
    const hasGanModels = state.assets && state.assets.gan.models.length > 0;
    $("classicRunButton").disabled = !hasInputs;
    $("fastRunButton").disabled = !hasInputs || !hasModels;
    $("ganRunButton").disabled = !hasInputs || !hasGanModels;
}

async function loadAssets(selectedInput = null) {
    const response = await fetch("/api/assets");
    state.assets = await response.json();

    $("inputFolder").textContent = state.assets.folders.input;
    $("outputFolder").textContent = state.assets.folders.output;

    fillSelect($("classicContent"), state.assets.classic.contents, "Upload an image first");
    fillSelect($("classicStyle"), state.assets.classic.styles, "Upload an image first");
    fillSelect($("fastContent"), state.assets.fast.contents, "Upload an image first");
    fillSelect($("fastModel"), state.assets.fast.models, "No model found");
    fillSelect($("ganContent"), state.assets.gan.contents, "Upload an image first");
    fillSelect($("ganModel"), state.assets.gan.models, "No GAN model found");

    selectIfExists($("classicContent"), selectedInput);
    selectIfExists($("classicStyle"), selectedInput);
    selectIfExists($("fastContent"), selectedInput);
    selectIfExists($("ganContent"), selectedInput);
    selectIfExists($("fastModel"), "good.model");
    selectIfExists($("ganModel"), "style_vangogh_pretrained");

    updatePreview($("classicContent"), $("classicContentPreview"));
    updatePreview($("classicStyle"), $("classicStylePreview"));
    updatePreview($("fastContent"), $("fastContentPreview"));
    updatePreview($("ganContent"), $("ganContentPreview"));
    updateRunButtons();
}

async function uploadInputImage() {
    const fileInput = $("inputImage");
    const file = fileInput.files[0];
    if (!file) {
        $("globalStatus").textContent = "Choose an image";
        return;
    }

    const payload = new FormData();
    payload.append("file", file);

    $("uploadButton").disabled = true;
    $("globalStatus").textContent = "Uploading";

    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: payload,
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Upload failed");
        }

        fileInput.value = "";
        $("globalStatus").textContent = "Ready";
        await loadAssets(result.file.name);
    } finally {
        $("uploadButton").disabled = false;
    }
}

async function cleanInputImages() {
    $("cleanInputButton").disabled = true;
    $("globalStatus").textContent = "Cleaning";

    try {
        const response = await fetch("/api/input/clean", {method: "POST"});
        if (!response.ok) {
            throw new Error(await response.text());
        }
        $("classicResult").classList.remove("ready");
        $("fastResult").classList.remove("ready");
        $("ganResult").classList.remove("ready");
        $("globalStatus").textContent = "Ready";
        await loadAssets();
    } finally {
        $("cleanInputButton").disabled = false;
    }
}

async function startJob(kind, payload) {
    const response = await fetch(`/api/run/${kind}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        throw new Error(await response.text());
    }

    return response.json();
}

function setRunning(kind, running) {
    const buttons = {
        classic: $("classicRunButton"),
        fast: $("fastRunButton"),
        gan: $("ganRunButton"),
    };
    const button = buttons[kind];
    button.disabled = running;
    if (!running) updateRunButtons();
}

function pollJob(kind, jobId) {
    const statusEl = $(`${kind}Status`);
    const logEl = $(`${kind}Log`);
    const resultEl = $(`${kind}Result`);

    clearInterval(state.timers[kind]);
    state.timers[kind] = setInterval(async () => {
        const response = await fetch(`/api/jobs/${jobId}`);
        const job = await response.json();

        statusEl.textContent = labelStatus(job.status);
        statusEl.dataset.state = job.status;
        $("globalStatus").textContent = job.status === "running" ? "Running" : "Ready";

        if (job.logs && job.logs.length) {
            logEl.textContent = job.logs.join("\n");
            logEl.classList.add("active");
            logEl.scrollTop = logEl.scrollHeight;
        }

        if (job.status === "done") {
            clearInterval(state.timers[kind]);
            setRunning(kind, false);
            resultEl.src = `${job.result_url}?t=${Date.now()}`;
            resultEl.classList.add("ready");
        }

        if (job.status === "error") {
            clearInterval(state.timers[kind]);
            setRunning(kind, false);
            statusEl.textContent = `Failed: ${job.error || "unknown error"}`;
            statusEl.dataset.state = "error";
        }
    }, 1800);
}

function labelStatus(status) {
    return {
        queued: "Queued",
        running: "Running",
        done: "Done",
        error: "Failed",
    }[status] || status;
}

$("classicContent").addEventListener("change", () => {
    updatePreview($("classicContent"), $("classicContentPreview"));
});

$("classicStyle").addEventListener("change", () => {
    updatePreview($("classicStyle"), $("classicStylePreview"));
});

$("fastContent").addEventListener("change", () => {
    updatePreview($("fastContent"), $("fastContentPreview"));
});

$("ganContent").addEventListener("change", () => {
    updatePreview($("ganContent"), $("ganContentPreview"));
});

$("uploadButton").addEventListener("click", () => {
    uploadInputImage().catch((error) => {
        $("globalStatus").textContent = "Upload failed";
        alert(error.message);
    });
});

$("cleanInputButton").addEventListener("click", () => {
    cleanInputImages().catch((error) => {
        $("globalStatus").textContent = "Clean failed";
        alert(error.message);
    });
});

$("classicForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    setRunning("classic", true);
    $("classicResult").classList.remove("ready");
    $("classicLog").classList.remove("active");
    $("classicStatus").textContent = "Submitting";
    $("classicStatus").dataset.state = "queued";

    try {
        const {jobId} = await startJob("classic", {
            content: $("classicContent").value,
            style: $("classicStyle").value,
            height: $("classicHeight").value,
            iterations: $("classicIterations").value,
            contentWeight: $("classicContentWeight").value,
            styleWeight: $("classicStyleWeight").value,
            tvWeight: $("classicTvWeight").value,
            initMethod: $("classicInit").value,
        });
        pollJob("classic", jobId);
    } catch (error) {
        $("classicStatus").textContent = `Failed: ${error.message}`;
        $("classicStatus").dataset.state = "error";
        setRunning("classic", false);
    }
});

$("fastForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    setRunning("fast", true);
    $("fastResult").classList.remove("ready");
    $("fastLog").classList.remove("active");
    $("fastStatus").textContent = "Submitting";
    $("fastStatus").dataset.state = "queued";

    try {
        const {jobId} = await startJob("fast", {
            content: $("fastContent").value,
            model: $("fastModel").value,
        });
        pollJob("fast", jobId);
    } catch (error) {
        $("fastStatus").textContent = `Failed: ${error.message}`;
        $("fastStatus").dataset.state = "error";
        setRunning("fast", false);
    }
});

$("ganForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    setRunning("gan", true);
    $("ganResult").classList.remove("ready");
    $("ganLog").classList.remove("active");
    $("ganStatus").textContent = "Submitting";
    $("ganStatus").dataset.state = "queued";

    try {
        const {jobId} = await startJob("gan", {
            content: $("ganContent").value,
            model: $("ganModel").value,
        });
        pollJob("gan", jobId);
    } catch (error) {
        $("ganStatus").textContent = `Failed: ${error.message}`;
        $("ganStatus").dataset.state = "error";
        setRunning("gan", false);
    }
});

loadAssets().catch((error) => {
    $("globalStatus").textContent = "Load failed";
    console.error(error);
});
