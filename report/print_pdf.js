// Render an HTML file to an A4 PDF with headless Chromium (Playwright).
// usage: node print_pdf.js <input.html> <output.pdf> "<footer text>"
const { chromium } = require('playwright');

(async () => {
  const [, , htmlPath, pdfPath, footer] = process.argv;
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + htmlPath, { waitUntil: 'load' });
  // wait until KaTeX has typeset every math span
  await page.waitForFunction(() => {
    const m = document.querySelectorAll('span.math');
    return m.length === 0 || document.querySelectorAll('span.math .katex').length === m.length;
  }, null, { timeout: 120000 });
  await page.evaluate(async () => { await document.fonts.ready; });
  const errors = await page.evaluate(() => Array.from(document.querySelectorAll('.katex-error')).map(e => e.title || e.textContent));
  if (errors.length) { console.error('KaTeX errors:', errors.slice(0, 10)); }
  await page.pdf({
    path: pdfPath, format: 'A4', printBackground: true, preferCSSPageSize: true,
    displayHeaderFooter: true,
    headerTemplate: '<div></div>',
    footerTemplate: '<div style="font-size:7.5px;width:100%;padding:0 16mm;display:flex;justify-content:space-between;color:#6b7280;font-family:Carlito,sans-serif">' +
      '<span>' + (footer || '') + '</span><span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>',
    margin: { top: '15mm', bottom: '17mm', left: '16mm', right: '16mm' },
  });
  await browser.close();
  console.log('pdf written', pdfPath, errors.length ? `(${errors.length} KaTeX errors)` : '');
})();
