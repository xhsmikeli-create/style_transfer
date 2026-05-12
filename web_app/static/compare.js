const compareState = {
    cases: [],
    selected: "mountain",
    view: "all",
};

const viewColumns = {
    all: ["original", "fastUkiyoe", "ganVanGogh", "ganUkiyoe"],
    ukiyoe: ["original", "fastUkiyoe", "ganUkiyoe"],
    gan: ["original", "ganVanGogh", "ganUkiyoe"],
};

const $ = (id) => document.getElementById(id);

function imageMarkup(image, modifier = "") {
    const missing = !image || !image.url;
    return `
        <figure class="compare-card ${modifier}">
            <div class="image-frame">
                ${missing ? `<span>缺少图片</span>` : `<img src="${image.url}" alt="${image.label}">`}
            </div>
            <figcaption>
                <strong>${image ? image.label : "-"}</strong>
                <span>${image ? image.name : ""}</span>
            </figcaption>
        </figure>
    `;
}

function renderTabs() {
    const tabs = $("caseTabs");
    tabs.innerHTML = compareState.cases.map((item) => `
        <button type="button"
            class="${item.id === compareState.selected ? "active" : ""}"
            data-case="${item.id}">
            ${item.label}
        </button>
    `).join("");

    tabs.querySelectorAll("button").forEach((button) => {
        button.addEventListener("click", () => {
            compareState.selected = button.dataset.case;
            render();
        });
    });
}

function renderFocus() {
    const current = compareState.cases.find((item) => item.id === compareState.selected);
    if (!current) return;

    $("caseTitle").textContent = current.label;
    $("caseDescription").textContent = current.description;

    const columns = viewColumns[compareState.view] || viewColumns.all;
    $("compareStrip").innerHTML = columns
        .map((key) => imageMarkup(current.images[key], key))
        .join("");
}

function renderGrid() {
    $("caseGrid").innerHTML = compareState.cases.map((item) => {
        return `
            <article class="case-row">
                <header>
                    <h3>${item.label}</h3>
                    <p>${item.description}</p>
                </header>
                <div class="case-images">
                    ${imageMarkup(item.images.original, "compact")}
                    ${imageMarkup(item.images.fastUkiyoe, "compact")}
                    ${imageMarkup(item.images.ganVanGogh, "compact")}
                    ${imageMarkup(item.images.ganUkiyoe, "compact")}
                </div>
            </article>
        `;
    }).join("");
}

function render() {
    renderTabs();
    renderFocus();
    renderGrid();
}

async function loadComparison() {
    const response = await fetch("/api/comparison");
    if (!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    compareState.cases = payload.cases || [];
    if (compareState.cases.length) {
        compareState.selected = compareState.cases[0].id;
    }
    $("compareStatus").textContent = "Ready";
    render();
}

$("viewMode").addEventListener("change", (event) => {
    compareState.view = event.target.value;
    renderFocus();
});

loadComparison().catch((error) => {
    $("compareStatus").textContent = "Load failed";
    $("caseDescription").textContent = error.message;
    console.error(error);
});
