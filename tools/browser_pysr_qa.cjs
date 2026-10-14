const {chromium}=require('playwright');
const path=require('node:path');
const fs=require('node:fs');
const root=path.resolve(__dirname,'..');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',args:['--enable-unsafe-swiftshader']});
 try {
 const page=await browser.newPage({viewport:{width:1520,height:1080}});
 await page.goto('http://127.0.0.1:8507');
 await page.getByText('Symbolic Lab',{exact:true}).click();
 await page.getByRole('button',{name:'Run symbolic search →'}).waitFor();
 await page.getByRole('combobox').first().click();
 await page.getByRole('option',{name:'PySR',exact:true}).click();
 await page.getByRole('tab',{name:'Expression constraints',exact:true}).click();
 await page.getByText('Nested-operator constraints',{exact:true}).waitFor();
 await page.getByRole('button',{name:'Run symbolic search →'}).click();
 await page.getByText('initializing',{exact:false}).first().waitFor({timeout:30000});
 await page.locator('[data-testid="stMain"]').evaluate(e=>e.scrollTop=0);
 await page.screenshot({path:path.join(root,'assets/screenshots/09-pysr-running.png'),fullPage:true});
 console.log('PYSR_RUNNING');
 await page.getByText('LIVE RESULT · PySR',{exact:false}).waitFor({timeout:240000});
 await page.locator('[data-testid="stMain"]').evaluate(e=>e.scrollTop=0);
 await page.screenshot({path:path.join(root,'assets/screenshots/10-pysr-result.png'),fullPage:true});
 fs.writeFileSync(path.join(root,'docs/pysr-browser-verification.json'),JSON.stringify({engine:'PySR',status:'complete',metrics:await page.locator('[data-testid="stMetric"]').allTextContents(),exception:await page.locator('[data-testid="stException"]').allTextContents()},null,2));
 console.log('PYSR_UI_RESULT_VERIFIED');
 } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
