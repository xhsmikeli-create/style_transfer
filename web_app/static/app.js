const state = {
    classicJob: null,
    fastJob: null,
    timers: {},
};

const $ = (id) => document.getElementById(id);

function fillSelect(select, items) {
    select.innerHTML = "";
    items.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.name;
        option.textContent = item.name;
        if (item.url) option.dataset.url = item.url;
        select.appendChild(option);
    });
}

function updatePreview(select, image) {
    const option = select.selectedOptions[0];
    if (!option || !option.dataset.url) {
        image.removeAttribute("src");
        return;
    }
    image.src = option.dataset.url;
}

async function loadAssets() {
    const response = await fetch("/api/assets");
    const assets = await response.json();

    fillSelect($("classicContent"), assets.classic.contents);
    fillSelect($("classicStyle"), assets.classic.styles);
    fillSelect($("fastContent"), assets.fast.contents);
    fillSelect($("fastModel"), assets.fast.models);

    setDefault($("classicContent"), "bear.jpg");
    setDefault($("classicStyle"), "candy.jpg");
    setDefault($("fastContent"), "golden_gate.jpg");
    setDefault($("fastModel"), "good.model");

    updatePreview($("classicContent"), $("classicContentPreview"));
    updatePreview($("classicStyle"), $("classicStylePreview"));
    updatePreview($("fastContent"), $("fastContentPreview"));
}

function setDefault(select, value) {
    const option = [...select.options].find((item) => item.value === value);
    if (option) select.value = value;
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
    const button = kind === "classic"
        ? $("classicForm").querySelector("button")
        : $("fastForm").querySelector("button");
    button.disabled = running;
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
            statusEl.textContent = `失败：${job.error || "未知错误"}`;
        }
    }, 1800);
}

function labelStatus(status) {
    return {
        queued: "排队中",
        running: "运行中",
        done: "已完成",
        error: "失败",
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

$("classicForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    setRunning("classic", true);
    $("classicResult").classList.remove("ready");
    $("classicStatus").textContent = "提交中";

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
        $("classicStatus").textContent = `失败：${error.message}`;
        setRunning("classic", false);
    }
});

$("fastForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    setRunning("fast", true);
    $("fastResult").classList.remove("ready");
    $("fastStatus").textContent = "提交中";

    try {
        const {jobId} = await startJob("fast", {
            content: $("fastContent").value,
            model: $("fastModel").value,
        });
        pollJob("fast", jobId);
    } catch (error) {
        $("fastStatus").textContent = `失败：${error.message}`;
        setRunning("fast", false);
    }
});

loadAssets().catch((error) => {
    $("globalStatus").textContent = "Load failed";
    console.error(error);
});
