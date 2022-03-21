import scrapy

class ToScrapeCSSSpider(scrapy.Spider):
    name = "bmwgroup"

    start_urls = [
        'https://www.bmwgroup.com/de.html',
    ]

    def parse(self, response):
        txt = response.css('p::text').getall()
        data = {}
        data['url'] = response.request.url
        data['text'] = txt
        yield data

        keys_to_consider = ['unternehmen', 
                            'verantwortung',
                            'innovation',
                            'marken',
                            'elektromobilitaet'
                            'news'
                            ]

        next_page_url = response.css("a::attr(href)").getall()
        for next_url in next_page_url:
            if next_url is not None:
                for key in keys_to_consider:
                    if key in next_url and \
                        'bmwgroup.com/de' in response.urljoin(next_url):
                        yield scrapy.Request(response.urljoin(next_url))

