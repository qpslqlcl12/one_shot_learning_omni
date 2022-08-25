import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle


#tf.debugging.set_log_device_placement(True)
start_time = time.time()

np.set_printoptions(precision=6, suppress=True)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


encoder_buffer=[]
TCAM_buffer=[]
#reverse_prediction=[]
TCAM_array=[]
backward_output=[]
support_data_set=[]
query_input=[]
#opt=tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.0)
opt=tf.keras.optimizers.Adam(learning_rate=1e-6)

#label for 4 way 
#way_0 = tf.Variable([1.,0.,0.,0.])
#way_1 = tf.Variable([0.,1.,0.,0.])
#way_2 = tf.Variable([0.,0.,1.,0.])
#way_3 = tf.Variable([0.,0.,0.,1.])
#way.append(way_0)
#way.append(way_1)
#way.append(way_2)
#way.append(way_3)

def load_img(fn):
    I=plt.imread(fn)
    I=np.array(I,dtype=bool)
    I=np.invert(I)
    I=I.astype('float32')
    I=I.flatten()
    return I

def testing_data_set_sampling():
    testing_img_dir = '../omniglot/python/images_evaluation'
    nalpha = 4 # number of alphabets to show
    alphabet_names = [a for a in os.listdir(testing_img_dir) if a[0] != '.'] # get folder names
    alphabet_names = random.sample(alphabet_names,nalpha) # choose random alphabets
    pick_query=random.randint(0,3)
   
    sampling_support_data_set=[]
    random_label_list=alphabet_names 
    for character in alphabet_names:
        character_id = random.randint(1,len(os.listdir(os.path.join(testing_img_dir,character))))
        string=str(character_id)
        img_char_dir = os.path.join(testing_img_dir,character,'character'+ string.zfill(2))
        support=random.randint(0,15)        
        support_set=os.listdir(img_char_dir)[support]  
        support_image=img_char_dir + '/' + support_set  
        testing_support_set=load_img(support_image)     
        
        sampling_support_data_set.append(testing_support_set)        
        
        if alphabet_names[pick_query] == character:
            query=random.randint(16,19)
            query_set=os.listdir(img_char_dir)[query]   
            query_image=img_char_dir + '/' + query_set
            query_label=character
    testing_query_set=load_img(query_image)
    sampling_query_way=random_label_list.index(query_label)
       

    return random_label_list,sampling_support_data_set,query_label,testing_query_set,sampling_query_way

def training_data_set_sampling():
    training_img_dir = '../omniglot/python/images_background'    
    nalpha = 4 # number of alphabets to show
    alphabet_names = [a for a in os.listdir(training_img_dir) if a[0] != '.'] # get folder names
    alphabet_names = random.sample(alphabet_names,nalpha) # choose random alphabets
    #print(alphabet_names) # 4 kinds of random alphabet
    sampling_support_data_set=[]
    #sampling_query_data_set=[]
    random_label_list=alphabet_names 
    #print(random_label_list)
    for character in alphabet_names:
        character_id = random.randint(1,len(os.listdir(os.path.join(training_img_dir,character))))
        string=str(character_id)
        img_char_dir = os.path.join(training_img_dir,character,'character'+ string.zfill(2))
        support=random.randint(0,15)
        #query=random.randint(16,19)
       
        support_set=os.listdir(img_char_dir)[support]
        #query_set=os.listdir(img_char_dir)[query]
       
        support_image=img_char_dir + '/' + support_set
        #query_image=img_char_dir + '/' + query_set
        
        training_support_set=load_img(support_image)
        #training_query_set=load_img(query_image)
        
        sampling_support_data_set.append(training_support_set)
        #sampling_query_data_set.append(training_query_set)
        #print(character)

    query=random.randint(16,19)
    query_set=os.listdir(img_char_dir)[query]
    # which is the last char id from upper
    query_image=img_char_dir + '/' + query_set
    training_query_set=load_img(query_image)
    sampling_query_data_set=training_query_set

    return random_label_list,sampling_support_data_set,query_set,sampling_query_data_set#, sampling_query_way



def data_preprocessing_for_testing(data_set, q_data_set):
    #empty_label = tf.Variable([0.,0.,0.,0.])
    data_buffer=[]    
    for i in (data_set):
        #print("way_buf",way_buf)
        #print("data_set",data_set)
        #sampled_data=tf.concat([i,j],0)
        #sampled_data=tf.expand_dims(sampled_data,0)
        sampled_data=tf.expand_dims(i,0)
        data_buffer.append(sampled_data)
    #sampled_q_data_set=tf.concat([empty_label,tf.squeeze(q_data_set)],0)    
    #sampled_q_data_set=tf.expand_dims(sampled_q_data_set,0)
    sampled_q_data_set=tf.expand_dims(q_data_set,0)

    return data_buffer,sampled_q_data_set

def data_preprocessing_for_training(data_set, q_data_set):
    #empty_label = tf.Variable([0.,0.,0.,0.])
    data_buffer=[]    
    for j in (data_set):
        #sampled_data=tf.concat([i,j],0)
        #sampled_data=tf.expand_dims(sampled_data,0)
        sampled_data=tf.expand_dims(j,0)
        data_buffer.append(sampled_data)

    query_data_buffer=[]
    #for l in q_data_set:
    #sampled_query_data=tf.concat([empty_label,q_data_set],0)
    #sampled_query_data=tf.expand_dims(sampled_query_data,0)
    sampled_query_data=tf.expand_dims(q_data_set,0)
    query_data_buffer=sampled_query_data  
        
        
    return data_buffer,query_data_buffer

def network_initializer():   
    ini_w1=tf.Variable(tf.random.uniform([11025,7000],-1,1), trainable=True)      
    ini_w2=tf.Variable(tf.random.uniform([7000,3000],-1,1), trainable=True)
    ini_w3=tf.Variable(tf.random.uniform([3000,1000],-1,1), trainable=True)
    ini_w4=tf.Variable(tf.random.uniform([1000,500],-1,1), trainable=True)
    ini_w5=tf.Variable(tf.random.uniform([500,100],-1,1), trainable=True)
    return ini_w1, ini_w2, ini_w3, ini_w4, ini_w5

def forward_pass(input,fw1,fw2,fw3,fw4,fw5):
    h1=tf.matmul(input,fw1)    
    h1=tf.keras.activations.sigmoid(h1)  
    #h1=tf.keras.activations.relu(h1)  

    h2=tf.matmul(h1,fw2)     
    h2=tf.keras.activations.sigmoid(h2) 
    #h2=tf.keras.activations.relu(h2) 
    h3=tf.matmul(h2,fw3)    
    h3=tf.keras.activations.sigmoid(h3)  
    #h3=tf.keras.activations.relu(h3)  
    h4=tf.matmul(h3,fw4)    
    h4=tf.keras.activations.sigmoid(h4) 
    #h4=tf.keras.activations.relu(h4) 
    out=tf.matmul(h4,fw5)
    out=tf.keras.activations.sigmoid(out)
    #out=tf.keras.activations.relu(out)
    return out,fw1,fw2,fw3,fw4,fw5


def contrastive_loss(encoded_query_data,cw1,cw2,cw3,cw4,cw5,query_way_buff):
    margin=5.5
    encoded_data,lw1,lw2,lw3,lw4,lw5 = forward_pass(encoded_query_data,cw1,cw2,cw3,cw4,cw5)
    retrieved_data, dist_btw_eQD_rD, retrieved_way, dist_btw_Q_TCAM = TCAM_retrieve(encoded_data)
    Dw=Euclidian_distance(encoded_data,retrieved_data)
    
    if query_way_buff==retrieved_way:
        similar = 1
        #print("similar")
    else:
        similar = 0
        #print("dissimilar")
    
    #similar=0
    loss_value=(similar)*(0.5)*(tf.square(Dw))+(1-similar)*(0.5)*tf.square((tf.math.maximum(0.,margin-Dw)))
    
    return loss_value, similar

def contrastive_loss_training(query_data_buff,cw1,cw2,cw3,cw4,cw5,support_data_buff,indexing):
    margin=5.5
    encoded_data,lw1,lw2,lw3,lw4,lw5 = forward_pass(query_data_buff,cw1,cw2,cw3,cw4,cw5)    
    Dw=Euclidian_distance(encoded_data,support_data_buff)
    
    if indexing==3:
        similar = 1
        #print("indexing = ",indexing,"similar")
    else:
        similar = 0#0
        #print("indexing = ",indexing,"dissimilar")
    
    #similar=0
    loss_value=(similar)*(0.5)*(tf.square(Dw))+(1-similar)*(0.5)*tf.square((tf.math.maximum(0.,margin-Dw)))
    
    return loss_value, similar


def grad(query_input,weight1,weight2,weight3,weight4,weight5,support_data_buff,indexing):
    with tf.GradientTape() as tape:
        loss,dummy=contrastive_loss_training(query_input,weight1,weight2,weight3,weight4,weight5,support_data_buff,indexing)        
    dw1,dw2, dw3,dw4,dw5 = tape.gradient(loss,[weight1,weight2,weight3,weight4,weight5])       
    return dw1,dw2,dw3,dw4,dw5, loss
 
def TCAM_store(encoded_data):
    TCAM_array.append(encoded_data)
    return 0
  
def TCAM_retrieve(encoded_query_data):
    dist1=[]
    dist2=[100.]     
    for stored_data in TCAM_array:        
        dist1.append(Euclidian_distance(encoded_query_data, stored_data))
    min_dist_value=min(dist1)
    min_dist_index=dist1.index(min_dist_value)
    return TCAM_array[min_dist_index], min_dist_value, min_dist_index, dist1

def Euclidian_distance(x,y):
    dist=tf.sqrt(tf.reduce_sum(tf.square(x-y)))    
    return dist


#training_data_set_sampling()
w1, w2, w3,w4,w5 = network_initializer()
"""
write_w1=open("weight_1.txt","rb")
w1=pickle.load(write_w1)
write_w1.close
write_w2=open("weight_2.txt","rb")
w2=pickle.load(write_w2)
write_w2.close
write_w3=open("weight_3.txt","rb")
w3=pickle.load(write_w3)
write_w3.close
write_w4=open("weight_4.txt","rb")
w4=pickle.load(write_w4)
write_w4.close
write_w5=open("weight_5.txt","rb")
w5=pickle.load(write_w5)
write_w5.close
"""
init_dists=[]
end_dists=[]
#sequence
def meta_testing(weight_buf1,weight_buf2,weight_buf3,weight_buf4,weight_buf5, save_trigger):
#meta_testing = 100 
    count=0
    answer=0
    repeat=0
    test_number=50   
    similarity_loss=0
    num_of_sim=0
    dissimilarity_loss=0
    num_of_dis=0
    for testing in range(test_number):
        count=count+1
        support_label_list,support_buff,query_label,query_data_buff,query_way = testing_data_set_sampling()
        support_data_set,query_data=data_preprocessing_for_testing(support_buff,query_data_buff) 
    #store support set
        for support_input,label in zip(support_data_set,support_label_list):
            encoder_buffer,w1,w2,w3,w4,w5 = forward_pass(support_input, weight_buf1,weight_buf2,weight_buf3,weight_buf4,weight_buf5)
            TCAM_store(encoder_buffer)
            
            if (save_trigger ==1) and testing == (test_number-1) :
                #qd=tf.slice(support_input,[0,3],[1,11025])
                qd=tf.reshape(support_input,[105,105])   
                plt.imshow(qd, cmap='gray')
                repeat=str(label)
                plt.savefig('./figures_init/'+repeat+'_support_input_init.png')

                edb=tf.reshape(encoder_buffer,[10,10])
                plt.imshow(edb, cmap='gray')
                plt.savefig('./figures_init/'+repeat+'_encoder_buffer_init.png')                

            if (save_trigger == 2) and testing == (test_number-1):
                #qd=tf.slice(support_input,[0,3],[1,11025])
                qd=tf.reshape(support_input,[105,105])   
                plt.imshow(qd, cmap='gray')
                repeat=str(label)
                plt.savefig('./figures_end/'+repeat+'_support_input_end.png')

                edb=tf.reshape(encoder_buffer,[10,10])
                plt.imshow(edb, cmap='gray')
                plt.savefig('./figures_end/'+repeat+'_encoder_buffer_end.png')
            


    #query
        encoder_buffer,w1,w2,w3,w4,w5 = forward_pass(query_data,w1,w2,w3,w4,w5)
        TCAM_buffer, distance_btw_query_mostsimilarTCAM, which_way, dist_btw_Q_TCAM = TCAM_retrieve(encoder_buffer)
        
        if (save_trigger ==1):
            if testing == test_number-1 or testing == test_number-10:
                init_dists.append(dist_btw_Q_TCAM)

        if (save_trigger == 2):
            if testing == test_number-1 or testing == test_number-10:
                print("init_dists:",init_dists)
                end_dists.append(dist_btw_Q_TCAM)
                print("final_dists:", end_dists)

        
        if (save_trigger ==1) and testing == (test_number-1) :
            #qd=tf.slice(query_data,[0,3],[1,11025])
            qd=tf.reshape(query_data,[105,105])   
            plt.imshow(qd, cmap='gray')
            repeat=str(label)
            plt.savefig('./figures_init/query_input_init.png')

            edb=tf.reshape(encoder_buffer,[10,10])
            plt.imshow(edb, cmap='gray')
            plt.savefig('./figures_init/encoded_Q_init.png')                

        if (save_trigger == 2) and testing == (test_number-1):
            #qd=tf.slice(query_data,[0,3],[1,11025])
            qd=tf.reshape(query_data,[105,105])   
            plt.imshow(qd, cmap='gray')
            repeat=str(label)
            plt.savefig('./figures_end/query_input_end.png')

            edb=tf.reshape(encoder_buffer,[10,10])
            plt.imshow(edb, cmap='gray')
            plt.savefig('./figures_end/encoded_Q_end.png')

        
        
        
        if(testing%100 == 0):
            print("testing_support_label",support_label_list)
            print("query_label", query_label)
            print("query_way",query_way)
            print("retrieved_way: ", which_way)  
            print("distances btw query & TCAM stored_data", dist_btw_Q_TCAM)

       
        if query_way == which_way:
            answer=answer+1
        accuracy=(answer/count)*100

        loss_buffer,similarity=contrastive_loss(query_data,w1,w2,w3,w4,w5,query_way)
        if similarity == 1:
            similarity_loss=similarity_loss+loss_buffer
            num_of_sim=num_of_sim+1
        elif similarity == 0:
            dissimilarity_loss=dissimilarity_loss+loss_buffer
            num_of_dis=num_of_dis+1
        TCAM_array.clear()

    similarity_loss=similarity_loss/(num_of_sim+1)
    dissimilarity_loss=dissimilarity_loss/(num_of_dis+1)

    return accuracy, similarity_loss, dissimilarity_loss

def meta_training(weight_buf1,weight_buf2,weight_buf3,weight_buf4,weight_buf5):
    train_number=100
    average_loss=0
    loss_sum=0
    
    for traning in range(train_number):
        TCAM_indexing=0
#meta_training, in training seq, query is set of all the ways.
        #support_label_list,support_buff,query_label,query_data_buff,query_way = training_data_set_sampling() 
        support_label_list,support_buff,query_label,query_data_buff = training_data_set_sampling() 
        support_data_set,query_data=data_preprocessing_for_training(support_buff,query_data_buff) 
        #print("support_set")
        #print(support_label_list)
        #print(support_data_set)
        #print(query_label)

     
        for support_input in support_data_set:    #4 times iter
            encoder_buffer,weight_buf1,weight_buf2,weight_buf3,weight_buf4,weight_buf5 = forward_pass(support_input,weight_buf1,weight_buf2,weight_buf3,weight_buf4,weight_buf5)
            TCAM_store(encoder_buffer)

        #for query_data_sample,query_way_list in zip(query_data,query_way):
        #grad_weight1,grad_weight2,grad_weight3,grad_weight4,grad_weight5,loss_value = grad(query_data_sample,weight_buf1,weight_buf2,weight_buf3,weight_buf4,weight_buf5,query_way_list)
        for encoded_support_data in TCAM_array:            
            grad_weight1,grad_weight2,grad_weight3,grad_weight4,grad_weight5,loss_value = grad(query_data,weight_buf1,weight_buf2,weight_buf3,weight_buf4,weight_buf5,encoded_support_data,TCAM_indexing)
            #print("loss:",loss_value)
            opt.apply_gradients(zip([grad_weight1,grad_weight2,grad_weight3,grad_weight4,grad_weight5],[weight_buf1,weight_buf2,weight_buf3,weight_buf4,weight_buf5]))
            TCAM_indexing=TCAM_indexing+1

        TCAM_array.clear()
    
    return weight_buf1,weight_buf2,weight_buf3,weight_buf4,weight_buf5


accuracy_list=[]
sim_loss_list=[]
dis_loss_list=[]
epoch=2000
max_acc=0
min_loss=0
trigger=0
for epoches in range(epoch):
    print('=======',epoches,'th epoch========')
    if epoches == 0:
        trigger=1
    if epoches == (epoch-1):
        trigger=2
        
        write_w1=open("weight_1.txt","wb")
        pickle.dump(w1,write_w1)
        write_w1.close
        write_w2=open("weight_2.txt","wb")
        pickle.dump(w2,write_w2)
        write_w2.close
        write_w3=open("weight_3.txt","wb")
        pickle.dump(w3,write_w3)
        write_w3.close
        write_w4=open("weight_4.txt","wb")
        pickle.dump(w4,write_w4)
        write_w4.close
        write_w5=open("weight_5.txt","wb")
        pickle.dump(w5,write_w5)
        write_w5.close
        
    acc,sim_loss,dis_loss=meta_testing(w1,w2,w3,w4,w5,trigger)
    trigger=0
    w1,w2,w3,w4,w5=meta_training(w1,w2,w3,w4,w5)
    """
    if epoches == 1:        
        gs=gridspec.GridSpec(4,4, hspace=0.2, wspace=0.2)
        plt.subplot(gs[0:3,0:3])
        plt.imshow(w1[:], cmap='gray', aspect ='auto')  
        plt.title("w1")
        plt.subplot(gs[0:3,3])
        plt.imshow(w2[:], cmap='gray', aspect ='auto')  
        plt.title("w2")
        plt.subplot(gs[-1,0:2])
        plt.imshow(w3[:], cmap='gray', aspect ='auto')  
        plt.title("w3")
        plt.subplot(gs[-1,2:3])
        plt.imshow(w4[:], cmap='gray', aspect ='auto')  
        plt.title("w4")
        plt.subplot(gs[-1,3])
        plt.imshow(w5[:], cmap='gray', aspect ='auto')  
        plt.title("w5")
        plt.savefig("w_init.png", dpi=300)

    if epoches == epoch-2:
        gs=gridspec.GridSpec(4,4, hspace=0.2, wspace=0.2)
        plt.subplot(gs[0:3,0:3])
        plt.imshow(w1[:], cmap='gray', aspect ='auto')  
        plt.title("w1")
        plt.subplot(gs[0:3,3])
        plt.imshow(w2[:], cmap='gray', aspect ='auto')  
        plt.title("w2")
        plt.subplot(gs[-1,0:2])
        plt.imshow(w3[:], cmap='gray', aspect ='auto')  
        plt.title("w3")
        plt.subplot(gs[-1,2:3])
        plt.imshow(w4[:], cmap='gray', aspect ='auto')  
        plt.title("w4")
        plt.subplot(gs[-1,3])
        plt.imshow(w5[:], cmap='gray', aspect ='auto')  
        plt.title("w5")
        plt.savefig("w_fin.png", dpi=3000)
    """
    sim_loss_list.append(sim_loss)
    dis_loss_list.append(dis_loss)
    print("accuracy:",acc,"%")
    accuracy_list.append(acc)
    if max_acc < acc:
        max_acc = acc
    #if min_loss > loss_buf:
        #min_loss = loss_buf
epoch=range(0,epoch)
plt.subplot(3,1,1)
plt.plot(epoch, accuracy_list)
plt.title('accuracy')
plt.subplot(3,1,2)
plt.plot(epoch, sim_loss_list)
plt.title('Test similarity loss')
plt.subplot(3,1,3)
plt.plot(epoch, dis_loss_list)
plt.title('Test dissimilarity loss')

plt.savefig('result.png', dpi=500)

print("maximum_accuracy", max_acc)
#print("minmum_loss", loss_buf)

print("Running time: {:.4f}min".format((time.time()-start_time)/60))
#end of code


