import { runSelfTest, type CheckResult } from "./gpu/selftest";

const out = document.getElementById("out")!;
const summary = document.getElementById("summary")!;
const results: CheckResult[] = [];

function render(result: CheckResult) {
  results.push(result);
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML =
    `<span class="${result.passed ? "pass" : "fail"}">${result.passed ? "PASS" : "FAIL"}</span> ` +
    `${result.name}<div class="detail">${result.detail}</div>`;
  out.appendChild(row);
  console.log(`[selftest] ${result.passed ? "PASS" : "FAIL"} ${result.name} — ${result.detail}`);
}

runSelfTest(render)
  .then((ok) => {
    const failed = results.filter((r) => !r.passed).length;
    summary.textContent = ok
      ? `ALL PASS (${results.length} checks)`
      : `${failed} FAILED of ${results.length}`;
    summary.className = ok ? "pass" : "fail";
    console.log(`[selftest] done: ${summary.textContent}`);
  })
  .catch((error) => {
    summary.textContent = `ERROR: ${error?.message ?? error}`;
    summary.className = "fail";
    console.error("[selftest]", error);
  });
