# Product Recommendation using Image Similarity
In this article we'll go over how to build a product recommendation system using image similarity of products that users have already clicked on. We'll upload a dataset, create a demo, and discuss how it works.

<img width="1237" alt="product-rec" src="https://github.com/user-attachments/assets/b180b1d6-2971-45af-b8c5-fbc8dbaae021">

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/tonyassi/product-recommendation)

## Download Code
Download the code from the [Github Repo](https://github.com/TonyAssi/product-recommendation)
```bash
git clone https://github.com/TonyAssi/product-recommendation.git 
```

## Dataset
We'll assume all you have is a folder of product images (1 image per product). We're going to create a Hugging Face dataset with the images and image embeddings. Image embeddings are generated with my [image embedding module](https://github.com/TonyAssi/HF-Embed-Images).

First you'll need to go into the *upload_ds.py* file from the repo you downloaded and change these 3 lines of code.
```python
img_folder = './img'
repo_name = 'tonyassi/product-images'
hf_token = 'YOUR_HF_TOKEN'
```

Then run the *upload_ds.py* file from the command line
```bash
python upload_ds.py
```

This should create 2 dataset repos in your Hugging Face profile. 

[tonyassi/product-images](https://huggingface.co/datasets/tonyassi/product-images) (images and ids)

[tonyassi/product-images-embeddings](https://huggingface.co/datasets/tonyassi/product-images-embeddings) (images, ids, and embeddings) This is the one we'll use for our product recommendation.

## Demo
Open the *app.py* file you orignial downloaded from Github and put in the repo of the dataset you just created.
```python
ds = load_dataset("tonyassi/finesse1-embeddings", split='train')
```

Go to Hugging Face and [create a new gradio space](https://huggingface.co/new-space?sdk=gradio). Upload *app.py* and *requirements.txt* files to the space. After your space is done building it should look like this [tonyassi/product-recommendation](https://huggingface.co/spaces/tonyassi/product-recommendation).

First a user will select their favorite product in the product gallery, 10 total. That product will get added to the prefence gallery. Then a new set of 10 products will be loaded into the product gallery. The top 5 products are the products most visually similar to the preference gallery products. The bottom 5 bottoms are selected at random. Products being loaded into the product gallery will never repeat, i.e. you'll never see the same products in the gallery until you refresh the page.

Congrats! Now you have a demo of a product recommendation system.

---

## Theory
Let's dig into how this work. You don't *need* to go through this section but it'll help you understand how it works and why I implemented the way I did.

### Image Embeddings
Image embeddings allow us to do image similarity search. But what are *image embeddings*? Simply put, image embeddings are a high level numerical representation of an image. They represent high level visual concepts to a computer, but to us they don't tell us much. Here is what they look like:
```python
[0.17519, -0.33182, -0.11692... 1.08443, -0.22943, 1.06595]
```

The image embeddings are generated with the [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) model. It is a great general purpose encoder model. In theory you could use a vision model fine-tuned on your dataset, but I have found this model works as good if not better fine-tuned models.

I have a very simple [image embedding module](https://github.com/TonyAssi/HF-Embed-Images) on Github which takes in a Hugging Face image dataset and creates a new image dataset with an embeddings column.

### Image Similarity
We perform image similarity by comparing the image embeddings of one image to another. This ends up being much faster and accurate then if we were to compare the pixel values of one image to another. 

In practice, to do the image embeddings comparisons we use [get_nearest_examples()](https://huggingface.co/docs/datasets/v2.7.1/en/package_reference/main_classes#datasets.Dataset.get_nearest_examples). This is a [FAISS](https://github.com/facebookresearch/faiss) function that is compatible with 🤗 Datasets.

First we add the FAISS index to the "embeddings" columns of our dataset.
```python
dataset_with_embeddings.add_faiss_index(column="embeddings")
```

Then we use the [get_nearest_examples()](https://huggingface.co/docs/datasets/v2.7.1/en/package_reference/main_classes#datasets.Dataset.get_nearest_examples) to find the image in the dataset which is most similar to the query image.
```python
scores, retrieved_examples = dataset_with_embeddings.get_nearest_examples(
    "embeddings", query_image_embedding, k=top_k
)
```

**scores** is a list of similarity scores, lower score means more similar.

**retrieved_examples** is a list of rows most similar to the query image.

If you want to read more about image similarity check out this [blog](https://huggingface.co/blog/image-similarity). Or check out this [notebook](https://colab.research.google.com/gist/sayakpaul/5b5b5a9deabd3c5d8cb5ef8c7b4bb536/image_similarity_faiss.ipynb) for code exmaples. Also you can my [image similarity module](https://github.com/TonyAssi/ImSim).

### Implementation
Let's go through the implementation of the [demo](https://huggingface.co/spaces/tonyassi/product-recommendation).

At first, 10 products are randomly selected and loaded into the product gallery. The reason for randomly selecting these initial products has to do with the [Cold Start Problem](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)), the system cannot draw any inference from users because there no preference data.

Once the user selects their favorite product it gets added to the prefence gallery. Then 10 new products from the dataset are loaded into the product gallery. Products being loaded into the product gallery will never repeat.

The top 5 products are the products most visually similar to the preference gallery products. A preference embedding is generated by taking the mean of all the image embeddings in the preference gallery. 

$$
\text{Preference Embedding} = \frac{1}{n} \sum_{i=1}^{n} \text{Image Embedding}_i
$$

This preference embedding is a vector representation of all the prefence images combined. This preference embedding is then used as the query embedding to fetch the most similar images in the dataset.

The bottom row of 5 products in the product gallery are selected randomly. The thinking behind this has to do with the cold start problem mentioned earlier. Let's say a user is looking for a yellow dress, but the initial set of products selected at random in the begining doesn't have a yellow dress. The user might select a red dress because they don't see the color they want. If all the products suggested to the user were similar then the user will keep being served red dresses and never get the opportunity to find their yellow dress. That's why some random products are suggested; to introduce various and a more broad search so that it doesn't get stuck in a local optima. We see a similar apporach in evolutionary algorithms where "DNA" is randomly mutated to introduce variation to the gene pool. We can play around with similar/random ratios but I found 5:5 was a nice balance.

### Performance
We can test the performance of our recommendation system to get some benchmarks. To do these benchmarks we assume there is a predefined target product in the dataset. The number of rounds it takes to find the target product is the score of the recommendation system. The lower score the better.

Let's first imagine a "brute force" recommendation system that randomly selects 10 products each round and *doesn't* consider the preferences of the user. The formula to find the number of rounds it'll take to find the target product on average is defined as follows:

- **N** total number of products
- **k** number of products shown per round

$$
E(X) \approx \frac{N}{2 \times k}
$$

If I have 500 products and I show 10 each time it'll take ~25 rounds on average to find the target product.

Now let's evalutate our recommendation system to see if we can do better. First we can randomly select a product from our dataset as our target product. Then we'll use the demo we made to try to find that target product. Do this 5-10 times to get an idea for the average number of rounds it takes to find the target product.

My dataset was 459 products and I showed 10 products per round. My average was 13 rounds to find the target product. Using the "brute force" method it would take ~23 rounds on average. This means our method is 43% faster than the brute force approach.

### Speed
Although speed and efficeiency wasn't the focus of this project, the code runs pretty fast. Each round is less than 1 second in terms of computation time. In the demo you might see it taking longer than a second, this is because the front end gradio code takes time to render the images.

### Improvement
There's plent of room for improvement like adding a text to search, optimizing the similar/random ratios of the product gallery, finetuning the ViT model and much more. For this article and demo I want to hyper-focus on the image similarity to understand how well it could work.

## About Me
Hello, my name is [Tony Assi](https://www.tonyassi.com/). I'm a designer based in Los Angeles. I have a background in software, fashion, and marketing. I currently work for an e-commerce fashion brand. Check out my [🤗 profile](https://huggingface.co/tonyassi) for more apps, models and datasets.

Feel free to send me an email at <tony.assi.media@gmail.com> with any questions, comments, business inquiries or job offers.
