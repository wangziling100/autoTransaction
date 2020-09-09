const {handler, defaultResponse, getBrowser} = require('../app')
describe('puppeteer browser', ()=>{
    let browser
    beforeAll(async()=>{
        browser = await getBrowser()
    }, 15000)
    afterAll(async()=>{
        await browser.close()
    }, 15000)
    test('it renders', async () => {
        const page = await browser.newPage();
        await page.goto('https://example.com');
        await page.screenshot({path: 'example.png'});
    })
    
}, 15000)

describe('static method', ()=>{
    test('response', ()=>{
        expect(defaultResponse()).toEqual({
            'statusCode': 200,
            'body': JSON.stringify({
                message: 'succeed'
            }),
            'headers': {
                "Access-Control-Allow-Origin":"*", 
                "Access-Control-Allow-Headers": "Content-Type",
            },
        })
    })
})
