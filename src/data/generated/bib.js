define({ entries : {
    "Bhatnagar2020Drone": {
        "abstract": "The application of drones has recently revolutionised the mapping of wetlands due to " +
            "their high spatial resolution and the flexibility in capturing images. In this study, the drone imagery " +
            "was used to map key vegetation communities in an Irish wetland, Clara Bog, for the spring season. " +
            "The mapping, carried out through image segmentation or semantic segmentation, was performed " +
            "using machine learning (ML) and deep learning (DL) algorithms. With the aim of identifying the " +
            "most appropriate, cost-efficient, and accurate segmentation method, multiple ML classifiers and DL " +
            "models were compared. Random forest (RF) was identified as the best pixel-based ML classifier, " +
            "which provided good accuracy (≈85%) when used in conjunction graph cut algorithm for image " +
            "segmentation. Amongst the DL networks, a convolutional neural network (CNN) architecture in a " +
            "transfer learning framework was utilised. A combination of ResNet50 and SegNet architecture gave " +
            "the best semantic segmentation results (≈90%). The high accuracy of DL networks was accompanied " +
            "with significantly larger labelled training dataset, computation time and hardware requirements " +
            "compared to ML classifiers with slightly lower accuracy. For specific applications such as wetland " +
            "mapping where networks are required to be trained for each different site, topography, season, " +
            "and other atmospheric conditions, ML classifiers proved to be a more pragmatic choice.",
        "author": "Bhatnagar, Saheba and Gill, Laurence and Ghosh, Bidisha",
        "doi": "10.3390/rs12162602",
        "journal": "Remote Sensing",
        "keywords": "type:Conference Paper, semantic segmentation; machine learning; random forest; deep learning; CNN.",
        "publisher": "MDPI",
        "series": "RS",
        "title": "Drone image segmentation using machine and deep learning for mapping raised bog vegetation communities",
        "url": "https://www.mdpi.com/2072-4292/12/16/2602",
        "volume": "12",
        "year": "2020",
    },
        "Shan2018Image": {
        "abstract": "The image is an important way for people to understand the world. How to make the computer have image " +
            "recognition function is the goal of image recognition research. In image recognition, image segmentation technology " +
            "is one of the important research directions. This paper uses gray-gradient maximum entropy method to extract " +
            "features from the image, uses K-mean method to classify the images, and uses average precision (AP) and intersection " +
            "over union (IU) evaluation methods to evaluate the results. The results show that the method of K-mean can achieve " +
            "image segmentation very well.",
        "author": "Pengfei Shan",
        "doi": "10.1186/s13640-018-0322-6",
        "journal": "EURASIP Journal on Image and Video Processing",
        "keywords": "type:Conference Paper, Image segmentation, K-mean, Clustering, Gray-level co-occurrence matrix (GLCM), Maximum entropy.",
        "publisher": "Springer",
        "series": "EJOIAVP",
        "title": "Image segmentation method based on K-mean algorithm",
        "url": "https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-018-0322-6",
        "type": "conference paper",
        "year": "2018"
    },
         "AIT SKOURT2018Lung": {
        "abstract": "Lung CT image segmentation is a necessary initial step for lung image analysis, it is a prerequisite step to provide an accurate lung CT image analysis such as lung cancer detection." +
             +"In this work, we propose a lung CT image segmentation using the U-net architecture, one of the most used architectures in deep learning for image segmentation. The architecture consists of a contracting path to extract high-level information and a symmetric expanding path that recovers the information needed. This network can be trained end-to-end from very few images and outperforms many methods. " +
             +"Experimental results show an accurate segmentation with 0.9502 Dice-Coefficient index.",
        "author": "Skourt, Brahim Ait and El Hassani, Abdelhamid and Majda, Aicha",
        "doi": "10.1016/j.procs.2018.01.104",
        "journal": "Procedia Computer Science",
        "keywords": "type:Conference Paper, Lung CT; Image Segmentation; Deep Learning; U-net.",
        "publisher": "Elsevier",
        "series": "PCS",
        "title": "Lung CT image segmentation using deep neural networks",
        "type": "conference paper",
        "volume": "127",
        "year": "2018"
    },

        "Müller2021MIScnn": {
        "abstract": "Background: The increased availability and usage of modern medical imaging induced a strong need for automatic " +
            "medical image segmentation. Still, current image segmentation platforms do not provide the required functionalities " +
            "for plain setup of medical image segmentation pipelines. Already implemented pipelines are commonly standalone " +
            "software, optimized on a specifc public data set. Therefore, this paper introduces the open-source Python library " +
            "MIScnn." +
            "Implementation: The aim of MIScnn is to provide an intuitive API allowing fast building of medical image segmentation pipelines including data I/O, preprocessing, data augmentation, patch-wise analysis, metrics, a library with stateof-the-art deep learning models and model utilization like training, prediction, as well as fully automatic evaluation " +
            "(e.g. cross-validation). Similarly, high confgurability and multiple open interfaces allow full pipeline customization." +
            "Results: Running a cross-validation with MIScnn on the Kidney Tumor Segmentation Challenge 2019 data set (multiclass semantic segmentation with 300 CT scans) resulted into a powerful predictor based on the standard 3D U-Net " +
            "model." +
            "Conclusions: With this experiment, we could show that the MIScnn framework enables researchers to rapidly set " +
            "up a complete medical image segmentation pipeline by using just a few lines of code. The source code for MIScnn is " +
            "available in the Git repository: https://github.com/frankkramer-lab/MIScnn..",
        "author": "Dominik Müller* and Frank Kramer",
        "doi": "10.1186/s12880-020-00543-7",
        "journal": "BMC medical imaging",
        "keywords": "type:Conference Paper, Medical image analysis, Computer aided diagnosis, Biomedical image segmentation, U-Net, Deep " + "learning, Open-source framework,",
        "publisher": "Springer",
        "series": "BMI",
        "title": "MIScnn: a framework for medical image segmentation with convolutional neural networks and deep learning",
        "type": "conference paper",
        "year": "2021"
    },
        "Huang2020Medical": {
        "abstract": " Pre-segmentation is known as a crucial step in medical image analysis. Many approaches have been proposed to " +
            "make improvement to both the quality and efficiency of segmentation. However, existing methods are lacking in robustness to " +
            "the variation in the edges and textures of the target. In order to address these drawbacks, a novel attention Gabor network " +
            "(AGnet) based on deep learning for medical image segmentation that is capable of automatically paying more attention to the " +
            "edge and consistently for improvement to the segmentation performance is proposed. The proposed model consists of two " +
            "components. The first one is to determine the approximate location of the organs of interest in the image using convolution " +
            "filters, and the other one is to highlight salient edge features intended for a specific segmentation task using Gabor filters. In " +
            "order to facilitate collaboration in between the two parts, a region attention mechanism based on Gabor maps is suggested. The " +
            "mechanism improved performance by learning to focus on the salient regions of the image that are useful for the authors' tasks. " +
            "As indicated by the experimental results, the AGnet is capable of enhancing the prediction performance while maintaining the " +
            "computational efficiency, which makes it comparable with other state-of-the-art approaches..",
        "author": "Huang, Shaoqiong and Huang, Mengxing and Zhang, Yu and Chen, Jing and Bhatti, Uzair",
        "doi": "10.1049/iet-ipr.2019.0772",
        "journal": "IET Image Processing",
        "keywords": "type:Conference Paper,",
        "publisher": "Wiley Online Librar",
        "series": "WOL",
        "title": "Medical image segmentation using deep learning with feature enhancement",
        "year": "2020"
    },
        "Saood2021COVID-19": {
        "abstract": "Background: Currently, there is an urgent need for efcient tools to assess the diagnosis of COVID-19 patients. In this " +
            "paper, we present feasible solutions for detecting and labeling infected tissues on CT lung images of such patients. " +
            "Two structurally-diferent deep learning techniques, SegNet and U-NET, are investigated for semantically segmenting infected tissue regions in CT lung images. " +
            "Methods: We propose to use two known deep learning networks, SegNet and U-NET, for image tissue classifcation. SegNet is characterized as a scene segmentation network and U-NET as a medical segmentation tool. " +
            "Both networks were exploited as binary segmentors to discriminate between infected and healthy lung tissue, also " +
            "as multi-class segmentors to learn the infection type on the lung. Each network is trained using seventy-two data " +
            "images, validated on ten images, and tested against the left eighteen images. Several statistical scores are calculated " +
            "for the results and tabulated accordingly." +
            "Results: The results show the superior ability of SegNet in classifying infected/non-infected tissues compared to " +
            "the other methods (with 0.95 mean accuracy), while the U-NET shows better results as a multi-class segmentor (with " +
            "0.91 mean accuracy)." +
            "Conclusion: Semantically segmenting CT scan images of COVID-19 patients is a crucial goal because it would not " +
            "only assist in disease diagnosis, also help in quantifying the severity of the illness, and hence, prioritize the population " +
            "treatment accordingly. We propose computer-based techniques that prove to be reliable as detectors for infected " +
            "tissue in lung CT scans. The availability of such a method in today’s pandemic would help automate, prioritize, fasten, " +
            "and broaden the treatment of COVID-19 patients globally.",
        "author": "Adnan Saood and Iyad Hatem",
        "doi": "10.1186/s12880-020-00529-5",
        "journal": "BMC Medical Imaging",
        "keywords": "type:Conference Paper, COVID-19, Pneumonia, SegNet, U-NET, Computerized tomography, Semantic segmentation.",
        "publisher": "Springer",
        "series": "BMI",
        "title": "COVID-19 lung CT image segmentation using deep learning methods: U-Net versus SegNet",
        "type": "conference paper",
        "year": "2021"
    },
        "ALI2020Brain": {
        "abstract": "Automated segmentation of brain tumour from multimodal MR images is pivotal for the " +
            "analysis and monitoring of disease progression. As gliomas are malignant and heterogeneous, efficient and " +
            "accurate segmentation techniques are used for the successful delineation of tumours into intra-tumoural " +
            "classes. Deep learning algorithms outperform on tasks of semantic segmentation as opposed to the more " +
            "conventional, context-based computer vision approaches. Extensively used for biomedical image segmentation, Convolutional Neural Networks have significantly improved the state-of-the-art accuracy on the task of " +
            "brain tumour segmentation. In this paper, we propose an ensemble of two segmentation networks: a 3D CNN " +
            "and a U-Net, in a significant yet straightforward combinative technique that results in better and accurate " +
            "predictions. Both models were trained separately on the BraTS-19 challenge dataset and evaluated to yield " +
            "segmentation maps which considerably differed from each other in terms of segmented tumour sub-regions " +
            "and were ensembled variably to achieve the final prediction. The suggested ensemble achieved dice scores of " +
            "0.750, 0.906 and 0.846 for enhancing tumour, whole tumour, and tumour core, respectively, on the validation " +
            "set, performing favourably in comparison to the state-of-the-art architectures currently available..",
        "author": "Ali, Mahnoor and Gilani, Syed Omer and Waris, Asim and Zafar, Kashan and Jamil, Mohsin",
        "doi": "10.1109/access.2020.3018160",
        "journal": "IEEE Access",
        "keywords": "type:Conference Paper, Deep learning, BraTS, medical imaging, segmentation, U-Net, CNN, ensembling.",
        "publisher": "Springer",
        "title": "Brain tumour image segmentation using deep networks",
        "type": "conference paper",
        "year": "2020"
    },
        "Sourati2019Intelligent": {
        "abstract": "Deep Convolutional Neural Networks (CNN) have " +
            "recently achieved superior performance at the task of medical " +
            "image segmentation compared to classic models. However, training a generalizable CNN requires a large amount of training " +
            "data, which is difficult, expensive and time consuming to obtain " +
            "in medical settings. Active Learning (AL) algorithms can facilitate " +
            "training CNN models by proposing a small number of the most " +
            "informative data samples to be annotated to achieve a rapid " +
            "increase in performance. We proposed a new active learning " +
            "method based on Fisher information (FI) for CNNs for the first " +
            "time. Using efficient backpropagation methods for computing " +
            "gradients together with a novel low-dimensional approximation " +
            "of FI enabled us to compute FI for CNNs with large number " +
            "of parameters. We evaluated the proposed method for brain " +
            "extraction with a patch-wise segmentation CNN model in two " +
            "different learning scenarios: universal active learning and active " +
            "semi-automatic segmentation. In both scenarios, an initial model " +
            "was obtained using labeled training subjects of a source data " +
            "set and the goal was to annotate a small subset of new samples " +
            "to build a model that performs well on the target subject(s). " +
            "The target data sets included images that differed from the " +
            "source data by either age group (e.g. newborns with different " +
            "image contrast) or underlying pathology that was not available " +
            "in the source data. In comparison to several recently proposed AL " +
            "methods and brain extraction baselines, the results showed that " +
            "FI-based AL outperformed the competing methods in improving " +
            "performance of the model after labeling a very small portion of " +
            "target data set (< 0.25%)..",
        "author": "Sourati, Jamshid and Gholipour, Ali and Dy, Jennifer G and Tomas-Fernandez, Xavier and Kurugol, Sila and Warfield, Simon K",
        "doi": "10.1109/tmi.2019.2907805",
        "journal": "IEEE Transactions on Medical Imaging",
        "keywords": "type:Conference Paper, Convolutional Neural Network, Active Learning," + "Fisher Information, Brain Extraction, Patch-wise Segmentation.",
        "publisher": "IEEE",
        "title": "Intelligent labeling based on fisher information for medical image segmentation using deep learning",
        "type": "conference paper",
        "year": "2019"
    },

        "Liu2015Semantic": {
        "abstract": "This paper addresses semantic image segmentation by" +
            "incorporating rich information into Markov Random Field" +
            "(MRF), including high-order relations and mixture of label" +
            "contexts. Unlike previous works that optimized MRFs" +
            "using iterative algorithm, we solve MRF by proposing a" +
            "Convolutional Neural Network (CNN), namely Deep Parsing Network (DPN)1" +
            ", which enables deterministic end-toend computation in a single forward pass. Specifically," +
            "DPN extends a contemporary CNN architecture to model" +
            "unary terms and additional layers are carefully devised to" +
            "approximate the mean field algorithm (MF) for pairwise" +
            "terms. It has several appealing properties. First, different" +
            "from the recent works that combined CNN and MRF, where" +
            "many iterations of MF were required for each training" +
            "image during back-propagation, DPN is able to achieve" +
            "high performance by approximating one iteration of MF." +
            "Second, DPN represents various types of pairwise terms," +
            "making many existing works as its special cases. Third," +
            "DPN makes MF easier to be parallelized and speeded up" +
            "in Graphical Processing Unit (GPU). DPN is thoroughly" +
            "evaluated on the PASCAL VOC 2012 dataset, where a single DPN model yields a new state-of-the-art segmentation" +
            "accuracy of 77.5%.",
        "author": "Liu, Ziwei and Li, Xiaoxiao and Luo, Ping and Loy, Chen-Change and Tang, Xiaoou",
        "doi": "10.1109/iccv.2015.162",
        "journal": "2015 IEEE International Conference on Computer Vision",
        "keywords": "type:Conference Paper,",
        "title": "Semantic Image Segmentation via Deep Parsing Network",
        "type": "conference paper",
        "year": "2015"
    },
        "Drews-Jr2021Underwater": {
        "abstract": "Image segmentation is an important step in many computer vision and image " +
            "processing algorithms. It is often adopted in tasks such as object detection, " +
            "classification, and tracking. The segmentation of underwater images is a challenging" +
            "problem as the water and particles present in the water scatter and absorb the light" +
            "rays. These effects make the application of traditional segmentation methods" +
            "cumbersome. Besides that, to use the state-of-the-art segmentation methods to face" +
            "this problem, which are based on deep learning, an underwater image segmentation" +
            "dataset must be proposed. So, in this paper, we develop a dataset of real underwater" +
            "images, and some other combinations using simulated data, to allow the training of" +
            "two of the best deep learning segmentation architectures, aiming to deal with" +
            "segmentation of underwater images in the wild. In addition to models trained in these" +
            "datasets, fine-tuning and image restoration strategies are explored too. To do a more" +
            "meaningful evaluation, all the models are compared in the testing set of real" +
            "underwater images. We show that methods obtain impressive results, mainly when" +
            "trained with our real dataset, comparing with manually segmented ground truth, even" +
            "using a relatively small number of labeled underwater training images..",
        "author": "Drews-Jr, Paulo and Souza, Isadora de and Maurell, Igor P and Protas, Eglen V and C. Botelho, Silvia S",
        "doi": "10.1186/s13173-021-00117-7",
        "journal": "Journal of the Brazilian Computer Society",
        "keywords": "type:Conference Paper, Underwater images, Segmentation, Deep learning, Sensing.",
        "publisher": "Springer",
        "title": "Underwater image segmentation in the wild using deep learning",
        "type": "conference paper",
        "year": "2021"
    },
}});