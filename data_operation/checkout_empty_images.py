import os,sys
import os.path as osp
import shutil

sys.path.append("/home/data1/yw/github_projects/personal_github/")
from code_aculat.utils.xml_process import analyze_xml
def check_empty_voc(voc_dir,mv_file=False):
    xml_dir=osp.join(voc_dir,'Annotations')
    xml_ls=os.listdir(xml_dir)

    emp_ls=[]
    for _ in xml_ls:
        class_ls,rect_ls=analyze_xml(osp.join(xml_dir,_))
        if len(class_ls)==0:
            emp_ls.append(osp.join(xml_dir,_))

    if mv_file:
        mv_voc(voc_dir,emp_ls)
    else:
        for _ in emp_ls:
            print('%s \n'%_)

def mv_voc(voc_dir,emp_ls,img_suf='.jpg'):
    emp_dir=osp.abspath(voc_dir)+'_emp'
    emp_anno=osp.join(emp_dir,'Annotations')
    emp_jpg=osp.join(emp_dir,'JPEGImages')

    os.makedirs(emp_anno,exist_ok=True)
    os.makedirs(emp_jpg,exist_ok=True)

    for _ in emp_ls:
        shutil.move(_,_.replace(voc_dir,emp_dir))

        jpg=_.replace('.xml',img_suf)
        jpg=jpg.replace('Annotations','JPEGImages')

        shutil.move(jpg,jpg.replace(voc_dir,emp_dir))


if __name__=="__main__":
    check_empty_voc('/home/data1/yw/mmdetection/data/water_detection/train_implmt',
                    mv_file=True
                    )
