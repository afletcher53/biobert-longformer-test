import urllib.request
import gzip
import os
import xml.etree.ElementTree as ET

def download_pubmed(year, month):
    base_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed24n{:04d}.xml.gz".format(year)
    filename = "pubmed22n{:04d}.xml.gz".format(year)
    urllib.request.urlretrieve(base_url, filename)
    return filename

def extract_abstracts(xml_file, output_file):
    with gzip.open(xml_file, 'rb') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        tree = ET.parse(f_in)
        root = tree.getroot()
        for article in root.findall('.//PubmedArticle'):
            abstract = article.find('.//AbstractText')
            if abstract is not None and abstract.text:
                f_out.write(abstract.text + '\n\n')

def prepare_pubmed_data(num_files=10):
    train_file = 'pubmed_train.txt'
    val_file = 'pubmed_val.txt'
    
    with open(train_file, 'w') as train, open(val_file, 'w') as val:
        for i in range(1, num_files + 1):
            filename = download_pubmed(i, 1)
            output = 'pubmed_extracted_{}.txt'.format(i)
            extract_abstracts(filename, output)
            
            # Split data: 90% train, 10% val
            with open(output, 'r') as f:
                lines = f.readlines()
                split = int(len(lines) * 0.9)
                train.writelines(lines[:split])
                val.writelines(lines[split:])
            
            os.remove(filename)
            os.remove(output)
    
    return train_file, val_file


train_file, val_file = prepare_pubmed_data(num_files=1)  # You can adjust the number of files