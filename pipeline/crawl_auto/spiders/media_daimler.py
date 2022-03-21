import scrapy

class ToScrapeCSSSpider(scrapy.Spider):
    name = "media_daimler"

    start_urls = [
        'https://media.daimler.com/marsMediaSite/en/instance/ko.xhtml?oid=4836258',
    ]

    def parse(self, response):
        txt = response.css('p::text').getall()
        data = {}
        data['url'] = response.request.url
        data['text'] = txt
        yield data

        keys_to_consider = ['marsMediaSite'
                            ]

        next_page_url = response.css("a::attr(href)").getall()
        for next_url in next_page_url:
            if next_url is not None:
                for key in keys_to_consider:
                    if key in next_url:
                        yield scrapy.Request(response.urljoin(next_url))
