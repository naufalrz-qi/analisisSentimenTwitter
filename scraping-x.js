const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const { Parser } = require('json2csv');

(async () => {
    const browser = await puppeteer.launch({
        headless: false, // Ganti ke true jika tidak ingin melihat browser berjalan
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
        ],
    });
    const page = await browser.newPage();

    // Set cookies untuk autentikasi
    await page.setCookie({
        name: 'auth_token',
        value: 'fdcb0a41e67681c13723ec560a5af934cac3aeb7',
        domain: '.x.com',
    });

    // Buka URL
    await page.goto('https://x.com/search?q=%23Indomie+lang%3Aid&src=recent_search_click', {
        waitUntil: 'networkidle2',
    });

    // Fungsi untuk menyimpan data ke CSV
    const saveToCSV = (data, filename) => {
        const fields = ['username', 'handle', 'content', 'time'];
        const opts = { fields };
        try {
            const parser = new Parser(opts);
            const csv = parser.parse(data);
            fs.appendFileSync(filename, csv + '\n', { flag: 'a' });
        } catch (err) {
            console.error(err);
        }
    };

    const filename = path.join(__dirname, 'tweets.csv');
    // Buat header CSV jika belum ada
    if (!fs.existsSync(filename)) {
        const header = 'username,handle,content,time\n';
        fs.writeFileSync(filename, header);
    }

    // Auto scrolling dan mengambil data
    const tweets = [];
    let previousHeight;

    while (tweets.length < 1000) {
        const newTweets = await page.evaluate(() => {
            const tweetElements = Array.from(document.querySelectorAll('[data-testid="tweet"]'));
            return tweetElements.map(tweet => ({
                username: tweet.querySelector('[data-testid="User-Name"]')?.innerText || 'N/A',
                handle: tweet.querySelector('[data-testid="User-Handle"]')?.innerText || 'N/A',
                content: tweet.querySelector('[data-testid="tweetText"]')?.innerText || 'N/A',
                time: tweet.querySelector('time') ? tweet.querySelector('time').getAttribute('datetime') : 'N/A',
            }));
        });

        tweets.push(...newTweets);
        saveToCSV(newTweets, filename);

        previousHeight = await page.evaluate('document.body.scrollHeight');
        await page.evaluate('window.scrollTo(0, document.body.scrollHeight)');
        await page.waitForFunction(`document.body.scrollHeight > ${previousHeight}`);

        // Gunakan setTimeout di dalam page.evaluate untuk menunggu beberapa detik sebelum scrolling lagi
        await page.evaluate(() => new Promise(resolve => setTimeout(resolve, 5000)));

        // Hentikan jika tidak ada tweet baru yang ditemukan
        if (newTweets.length === 0) break;
    }

    console.log('Scraping selesai, data disimpan dalam tweets.csv');
    await browser.close();
})();
