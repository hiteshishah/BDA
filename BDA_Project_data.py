import json

def main():
    restaurant_stars = {}
    with open('business.json', 'r') as file:
        for line in file:
            business = json.loads(line)
            for key, value in business.items():
                if key == "categories":
                    if "Mexican" in value:
                        restaurant_stars[business.get("business_id")] = business.get("stars")

    restaurant_ids = list(restaurant_stars.keys())

    restaurant_reviews = {}
    with open('review.json', 'r', encoding='utf8') as file:
        for line in file:
            business = json.loads(line)
            for key, value in business.items():
                if key == "business_id":
                    if value in restaurant_ids:
                        if value in restaurant_reviews.keys():
                            restaurant_reviews.get(value).append(business.get("text"))
                        else:
                            restaurant_reviews[business.get("business_id")] = [business.get("text")]

    restaurants = {k: [restaurant_stars[k], restaurant_reviews[k]] for k in restaurant_stars}

    with open('restaurant.json', 'w') as file:
        file.write(json.dumps(restaurants))

main()