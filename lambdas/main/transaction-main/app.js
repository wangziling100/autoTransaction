const puppeteer = require('puppeteer');


exports.handler = async (event, context) =>{
    const body = JSON.parse(event.body)
    const command = body.command
    switch(command){
        case 'stop': return defaultResponse();
        default: {
            let browser = await getBrowser()
            await browser.close()
            break
        } 
    }
    return defaultResponse()
}

exports.process = process
async function process(driver){
    console.log('here1')
    await driver.get('www.google.de')
    //const screenshot = await driver.takeScreenshot()
    //console.log(screenshot, 'screenshot')
}
exports.getBrowser = getBrowser
async function getBrowser(){
    const browser = await puppeteer.launch({args:['--no-sandbox']});
    return browser
}

exports.setResponse=setResponse
function setResponse(options){
    const statusCode = options.statusCode
    delete options.statusCode
    const response = {
        'statusCode': statusCode,
        'body': JSON.stringify(options),
        'headers': {"Access-Control-Allow-Origin":"*", "Access-Control-Allow-Headers": "Content-Type",},
    }
    return response
}

exports.defaultResponse=defaultResponse
function defaultResponse(){
    const options = {
        statusCode: 200,
        message: 'succeed',
    }
    return setResponse(options)
}
/*
const defaultEvent = {
    body: JSON.stringify({
        message: 'test'
    })
}
exports.handler(defaultEvent)
*/