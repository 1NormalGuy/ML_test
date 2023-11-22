import normalization as nm
import os


# 获得大词典并转储为allDic.txt
def get_dic(datasetChoice):
    allDic = nm.get_dictionary(datasetChoice + '/train')
    nm.dic_to_txt(allDic, datasetChoice + '/allDic.txt')

# 文档转储为向量（按索引排序）
def transfer_raw_data_to_vectors(datasetChoice, setChoice, subsetChoice):
    allDic = nm.txt_to_dic(datasetChoice + '/allDic.txt')
    pathLoadChoice = datasetChoice + '/' + setChoice
    pathStoreChoice = datasetChoice + '/v_' + setChoice
    for dirname in os.listdir(pathLoadChoice):
        dirPath = pathLoadChoice + '/' + dirname
        dirPathVec = pathStoreChoice + '/' + dirname
        if not os.path.exists(dirPathVec):
            os.makedirs(dirPathVec)
        print('loading ' + dirPath + '...')
        print('loading ' + dirPathVec + '...')
        for filename in os.listdir(dirPath):
            sorted_vec = nm.get_sorted_vector(dirPath + '/' + filename, allDic)
            nm.dic_to_txt(sorted_vec, dirPathVec + '/' + filename)

datasetChoice = 'dataof2'
setChoice = 'train'
get_dic(datasetChoice)
transfer_raw_data_to_vectors(datasetChoice,setChoice,'test')
