import scrapy
from scrapy.crawler import CrawlerProcess

class MyprojectItem(scrapy.Item):
    # define the fields for your item here
    content = scrapy.Field()

class AllContentSpider(scrapy.Spider):
    name = 'all_content'
    allowed_domains = ['https://www.velo.be/nl', 'https://www.google.com/mymaps/viewer?mid=1gN1ka1mBu2aVOUlYX3ahSaOFi5I&hl=en_US', 'https://www.velo.be/nl/fietsverhuur', 'https://www.companyweb.be/nl/0453069875/velo', 'https://www.velo.be/nl/contact', 'https://be.linkedin.com/company/velo-vzw']  # Change to the domain you want to scrape
    start_urls = ['https://www.velo.be/nl', 'https://www.google.com/mymaps/viewer?mid=1gN1ka1mBu2aVOUlYX3ahSaOFi5I&hl=en_US', 'https://www.velo.be/nl/fietsverhuur', 'https://www.companyweb.be/nl/0453069875/velo', 'https://www.velo.be/nl/contact', 'https://be.linkedin.com/company/velo-vzw']  # Change to the domain you want to scrape

    custom_settings = {
        'FEED_FORMAT': 'json',  # or 'csv', 'xml', etc.
        'FEED_URI': 'output.json'  # Output filename
    }

    def parse(self, response):
        item = MyprojectItem()
        raw_content = response.xpath('//body//text()').extract()  # Extract all text within the body tag
        cleaned_content = [text.strip() for text in raw_content if text.strip()]  # Remove unnecessary whitespaces and blank lines
        item['content'] = cleaned_content
        yield item

        # To follow links and scrape content from them as well, uncomment the following lines:
        # for href in response.css('a::attr(href)').getall():
        #     yield response.follow(href, self.parse)

# Main driver
if __name__ == "__main__":
    # Run the spider
    process = CrawlerProcess()
    process.crawl(AllContentSpider)
    process.start()
