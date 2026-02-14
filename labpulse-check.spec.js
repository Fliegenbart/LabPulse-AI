import { test, expect } from '@playwright/test'

test('dashboard render marker', async ({ page }) => {
  await page.goto('http://5.9.106.75:8501/?open=dashboard', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1200);
  const txt = await page.locator('body').innerText();
  console.log('---TXT_START---');
  console.log(txt.slice(0, 5000));
  console.log('---TXT_END---');
  const hasHome = txt.includes('Home') || txt.includes('🏠');
  const hasLanding = txt.includes('Reagenzbedarf') || txt.includes('Alles in einem Dashboard.');
  console.log('HAS_HOME', hasHome);
  console.log('HAS_LANDING', hasLanding);
  expect([hasHome, hasLanding]).toBeTruthy();
});
