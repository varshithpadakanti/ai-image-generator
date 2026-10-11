const f = document.getElementById("f"), go = document.getElementById("go");
const status = document.getElementById("status"), out = document.getElementById("out");
const img = document.getElementById("img"), dl = document.getElementById("dl");

f.addEventListener("submit", async (e) => {
  e.preventDefault();
  go.disabled = true; out.hidden = true;
  status.textContent = "Generating. This usually takes 10 to 40 seconds.";
  try {
    const res = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: document.getElementById("prompt").value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong.");
    img.src = data.image; dl.href = data.image; out.hidden = false;
    status.textContent = "";
  } catch (err) {
    status.textContent = err.message;
  } finally {
    go.disabled = false;
  }
});
