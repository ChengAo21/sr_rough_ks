// A fresh headless browser, used only to verify this locally developed app.
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const out = path.join(root, 'assets', 'screenshots');
fs.mkdirSync(out, { recursive: true });
(async () => {
  const browser = await chromium.launch({headless:true, executablePath:'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',args:['--enable-unsafe-swiftshader']});
  try {
  const page = await browser.newPage({viewport:{width:1520,height:1080},deviceScaleFactor:1});
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  async function capture(name, resetScroll=true) {
    const text=await page.locator('[data-testid="stAppViewContainer"]').innerText();
    if (/[\u3400-\u4dbf\u4e00-\u9fff]/u.test(text)) throw new Error('Untranslated text found in '+name);
    if (resetScroll) await page.locator('[data-testid="stMain"]').evaluate(el=>{el.scrollTop=0;});
    await page.screenshot({path:path.join(out,name),fullPage:true});
  }
  await page.goto('http://127.0.0.1:8507', {waitUntil:'domcontentloaded'});
  await page.getByText('drag modelling',{exact:false}).first().waitFor({timeout:40000});
  await page.locator('.js-plotly-plot').first().waitFor();
  await capture('01-overview.png');
  console.log('OVERVIEW_READY');
  await page.getByText('Symbolic Lab',{exact:true}).click();
  await page.getByRole('button',{name:'Run symbolic search →'}).waitFor();
  await page.getByRole('combobox').first().click();
  await page.getByRole('option',{name:'Python GP',exact:true}).click();
  await page.getByRole('tab',{name:'Expression constraints',exact:true}).click();
  await page.getByText('Nested-operator constraints',{exact:true}).waitFor();
  await capture('02-symbolic-controls.png');
  await page.getByText('Limits apply to the selected operators. The default table is the original nested_constraints configuration.',{exact:true}).scrollIntoViewIfNeeded();
  await capture('12-nested-constraints.png',false);
  await page.getByRole('button',{name:'Run symbolic search →'}).click();
  await page.getByText('LIVE RESULT ·',{exact:false}).waitFor({timeout:90000});
  await capture('03-symbolic-result.png');
  console.log('SYMBOLIC_RUN_COMPLETE');
  await page.getByRole('combobox').nth(1).click();
  await page.getByRole('option',{name:'Paper settings',exact:true}).click();
  await page.getByRole('tab',{name:'Search budget',exact:true}).click();
  await capture('11-paper-settings.png');
  await page.getByText('PCE Studio',{exact:true}).click();
  await page.getByRole('button',{name:'Fit PCE & analyze →'}).waitFor();
  await page.getByText('DISPLAYED RUN ·',{exact:false}).waitFor();
  const frame = page.frameLocator('iframe').first();
  await frame.getByText('▶ Play',{exact:true}).waitFor({timeout:30000});
  await capture('04-pce-studio.png');
  await page.getByRole('button',{name:'Fit PCE & analyze →'}).click();
  await page.getByText('Surrogate ready ·',{exact:false}).waitFor({timeout:40000});
  await frame.getByText('▶ Play',{exact:true}).waitFor({timeout:30000});
  await frame.getByText('▶ Play',{exact:true}).click();
  await capture('05-pce-animation.png');
  await page.getByRole('tab',{name:'Sensitivity',exact:true}).click();
  await capture('06-pce-sensitivity.png');
  console.log('PCE_RUN_AND_PLAY_VERIFIED');
  await page.getByText('Model Explorer',{exact:true}).click();
  await page.getByText('Subfunctions and pairwise interactions',{exact:true}).waitFor();
  await capture('07-model-explorer.png');
  await page.getByText('Research & Cite',{exact:true}).click();
  await page.getByText('Keep the provenance visible.',{exact:true}).waitFor();
  await capture('08-references.png');
  const exceptions=await page.locator('[data-testid="stException"]').allTextContents();
  fs.writeFileSync(path.join(root,'docs','browser-verification.json'),JSON.stringify({url:'http://127.0.0.1:8507',errors,exceptions,steps:['Pairwise-product overview loaded','Nesting and weighted-complexity controls displayed','Constrained Python GP searched','Paper settings restored','PCE fitted','Projected pairwise PCE animation played','Sensitivity displayed','Model explorer pairwise products loaded','Citations loaded']},null,2));
  console.log(JSON.stringify({errors,exceptions}));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
