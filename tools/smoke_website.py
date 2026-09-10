#!/usr/bin/env python3
"""UI smoke tests on original public display data, not scientific recertification.

Install Playwright and Google Chrome (or `playwright install chrome`). Official
Chrome includes the H.264 codecs used by the original, unmodified MP4 assets.
"""
import argparse, functools, http.server, json, threading
from pathlib import Path
from playwright.sync_api import sync_playwright

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_): pass
    def copyfile(self, source, outputfile):
        try:
            super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browsers legitimately cancel video downloads on reload/seek.

def require(value, message):
    if not value: raise AssertionError(message)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=Path, default=Path('website'))
    parser.add_argument('--out', type=Path, default=Path('website-tests.json'))
    parser.add_argument('--online')
    args = parser.parse_args()
    root = args.site.resolve()
    datasets = {'objective_repair_20260909': json.loads((root/'assets/evidence-objective.json').read_text()),
                'v3_20260909': json.loads((root/'assets/evidence.json').read_text())}
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = args.online.rstrip('/')+'/' if args.online else f'http://127.0.0.1:{server.server_port}/'
    report = {'scope': 'Browser UI smoke test, not optimization, dynamics or exact arithmetic replay', 'checks': [], 'javascript_errors': [], 'http_errors': [], 'passed': False}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='chrome', headless=True, args=['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
            report['browser'] = {'channel':'chrome', 'version':browser.version}
            page = browser.new_page(viewport={'width':1366, 'height':900}, reduced_motion='reduce')
            page.set_default_timeout(20000)
            page.on('pageerror', lambda error: report['javascript_errors'].append(str(error)))
            page.on('response', lambda response: report['http_errors'].append({'url':response.url, 'status':response.status}) if response.url.startswith(base) and response.status >= 400 and not response.url.endswith('favicon.ico') else None)
            page.goto(base, wait_until='domcontentloaded')
            page.wait_for_function("document.querySelectorAll('#scene-select option').length===20 && document.querySelector('#viewer-status').hidden")
            require(page.locator('#scene-canvas canvas').count()==1, 'Expected one canvas')
            require(page.locator('.hero-pause').count()==1, 'Expected one generated hero control')
            require(page.locator('#viewer-fallback').is_hidden(), 'WebGL must be active')
            require(page.locator('#campaign-select option').count()==2, 'Expected two campaigns')
            report['checks'].append('WebGL active; single canvas; no duplicate controls; two campaigns')
            for campaign, dataset in datasets.items():
                page.select_option('#campaign-select', campaign)
                for scene in dataset['scenes']:
                    page.select_option('#scene-select', str(scene['id']))
                    require(str(scene['id']) in page.locator('#scene-caption').inner_text(), 'Scene caption')
                    require(scene['runId'] in page.locator('#record-label').inner_text(), 'Run identity')
                    failed = scene['status']=='ROOT_FAILURE'
                    require(page.locator('#failure-message').is_visible()==failed, 'Historical failure visibility')
                    require(page.locator('#play-button').is_disabled()==failed, 'Playback availability')
                    if not failed:
                        require(page.locator('#error-chart path').count()>0, 'Error telemetry chart')
                        require(page.locator('#clearance-chart path').count()>0, 'Clearance telemetry chart')
                    report['checks'].append(f"{campaign}/{scene['id']}: identity, status, playback and telemetry")
            page.select_option('#campaign-select', 'objective_repair_20260909')
            page.select_option('#scene-select', '41015')
            page.locator('#scene-canvas').scroll_into_view_if_needed()
            for stage in range(4):
                selector=f'[data-stage="{stage}"]'
                page.locator(selector).click()
                require(page.locator(selector).get_attribute('aria-pressed')=='true', 'Stage selection')
            for view in ['top', 'side', 'perspective']:
                selector=f'[data-view="{view}"]'
                page.locator(selector).click()
                require(page.locator(selector).get_attribute('aria-pressed')=='true', 'Camera selection')
            for layer in ['projected', 'recovered', 'tracked', 'envelope']:
                page.locator('#layer-'+layer).uncheck()
                page.locator('#layer-'+layer).check()
            page.select_option('#play-speed', '2')
            page.locator('#play-button').click()
            page.wait_for_function("parseFloat(document.querySelector('#play-time').textContent)>0.1")
            page.locator('#play-button').click()
            page.locator('#timeline').evaluate("e=>{e.value='3';e.dispatchEvent(new Event('input',{bubbles:true}));}")
            require(page.locator('#play-time').inner_text()=='3.00 s', 'Timeline seek')
            page.locator('#restart-button').click()
            require(page.locator('#play-time').inner_text()=='0.00 s', 'Restart')
            report['checks'].append('Four stages, three camera presets, four layer toggles, speed, playback, seek and restart')
            require(page.locator('#energy-chart rect').count()>0, 'Historical histogram')
            page.locator('video').evaluate_all('(videos)=>videos.forEach(v=>{v.muted=true;v.preload="auto";v.load();})')
            try:
                page.wait_for_function('[...document.querySelectorAll("video")].every(v=>v.readyState>=1 && v.videoWidth>0)', timeout=45000)
                page.locator('video').evaluate_all('(videos)=>Promise.all(videos.map(v=>v.play()))')
                page.wait_for_function('[...document.querySelectorAll("video")].every(v=>v.currentTime>0.1 && v.readyState>=2)', timeout=30000)
                page.locator('video').evaluate_all('(videos)=>videos.forEach(v=>v.pause())')
            finally:
                report['video_diagnostics'] = page.locator('video').evaluate_all('(videos)=>videos.map(v=>({src:v.currentSrc,readyState:v.readyState,networkState:v.networkState,width:v.videoWidth,height:v.videoHeight,time:v.currentTime,error:v.error?{code:v.error.code,message:v.error.message}:null,h264:v.canPlayType(\'video/mp4; codecs="avc1.640028"\')}))')
                print('VIDEO_DIAGNOSTICS', json.dumps(report['video_diagnostics']), flush=True)
            report['checks'].append('Historical histogram; all three original H264 videos decode and play')
            page.set_viewport_size({'width':390, 'height':844})
            page.wait_for_timeout(300)
            require(page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+2'), 'Mobile horizontal overflow')
            report['checks'].append('390px mobile viewport without horizontal overflow')
            require(not report['javascript_errors'], 'JavaScript errors: '+str(report['javascript_errors']))
            require(not report['http_errors'], 'Missing public assets: '+str(report['http_errors']))
            report.update(passed=True, total_checks=len(report['checks']))
            browser.close()
    finally:
        server.shutdown(); server.server_close(); thread.join()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__=='__main__': main()
