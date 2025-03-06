from flask import Flask,render_template,redirect,url_for,request,session
import pymysql
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import nltk # Natural Language Toolkit
nltk.download('stopwords')
from sklearn.neighbors import NearestNeighbors
app=Flask(__name__)
app.config['SECRET_KEY']='f9bf78b9a18ce6d46a0cd2b0b86df9da'
db = pymysql.connect(host='localhost',user='root',password='',db='movierecommendation',port=3306)
cursor=db.cursor()

@app.route("/")
def index():
    return render_template('index.html')
@app.route("/adminlogin",methods=["POST","GET"])
def adminlogin():
    if request.method=="POST":
        name=request.form["name"]
        pwd=request.form['pwd']
        print(pwd,name)
        if name=="Admin" and pwd=="Admin":
            return render_template("adminhome.html")
    return render_template('adminlogin.html')
@app.route('/view_users')
def view_users():
    sql="select * from users"
    results=pd.read_sql_query(sql,db)
    print(results)
    results.drop(["cpwd"],axis=1,inplace=True)
    results["delete"]="delete"
    return render_template("view_users.html",row_val=results.values.tolist())

@app.route("/deletedfunction/<s1>")
def deletedfunction(s1=0):
    print(s1)
    sql="delete from users where slno='%s'"%(s1)
    cursor.execute(sql)
    db.commit()
    return redirect(url_for('view_users'))





@app.route('/view_reviews')
def view_reviews():
    sql="select * from reviews"
    results=pd.read_sql_query(sql,db)
    print(results)
    return render_template("view_reviews.html",row_val=results.values.tolist())

@app.route("/View_history")
def View_history():
    sql="select * from history"
    data=pd.read_sql_query(sql,db)
    print(data)
    return render_template("View_history.html",row_val=data.values.tolist())

@app.route("/userreg",methods=["POST","GET"])
def userreg():
    if request.method=="POST":
        name=request.form["name"]
        email=request.form["email"]
        gender = request.form["gender"]
        mobile = request.form["number"]
        pwd=request.form["pwd"]
        cpwd=request.form["cpwd"]
        if pwd==cpwd:
            sql = "insert into users (name,email,gender,number,pwd,cpwd) values (%s,%s,%s,%s,%s,%s)"
            values = (name,email,gender,mobile,pwd,cpwd)

            cursor.execute(sql, values)
            db.commit()
            return render_template("userreg.html",ms="success")
        else:
            return render_template("userreg.html",ms="fail")
    return render_template("userreg.html")

movies = pd.read_csv("C:/Users/HP/OneDrive/Desktop/TO project/TK13041/TK13041/DATASET/movies.csv")
ratings = pd.read_csv("C:/Users/HP/OneDrive/Desktop/TO project/TK13041/TK13041/DATASET/ratings.csv")
final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')

final_dataset.fillna(0, inplace=True)








no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

sample = np.array([[0, 0, 3, 0, 0], [4, 0, 0, 0, 2], [0, 0, 0, 0, 1]])
sparsity = 1.0 - (np.count_nonzero(sample) / float(sample.size))

csr_sample = csr_matrix(sample)
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)
def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_reccomend + 1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                   key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
        df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_reccomend + 1))
        return df
    else:
        return "No movies found. Please check your input"

@app.route("/userlogin",methods=['POST','GET'])
def userlogin():
    if request.method=='POST':
        name = request.form["name"]
        pwd = request.form["pwd"]
        print(name,pwd)
        sql="select * from users where name=%s and pwd=%s "
        val=(name,pwd)
        X=cursor.execute(sql,val)
        Results=cursor.fetchall()
       





 print(X)
        print(Results)
        if X>0:
            session["userid"]=Results[0][0]
            session["username"]=Results[0][1]
            return render_template("userhome.html",msg="sucess")
        else:
            print("5555555555555")
            return render_template("userlogin.html",mfg="not found")
    return render_template("userlogin.html")

@app.route("/Moviewname",methods=["POST","GET"])
def Moviewname():
    if request.method=="POST":
        name=request.form["moviewname"]
        Final_results=get_movie_recommendation(name)
        F1=pd.DataFrame(Final_results)
        print(F1)
        F2=F1.drop(["Distance"],axis=1)
        F2["Review"]="Review"
        F2.values.tolist()
        return render_template("Moviewnamesuggestion.html",res=F2.values.tolist())
    return render_template('Moviewname.html')

@app.route("/enterreview/<s1>",methods=["POST","GET"])
def enterreview(s1=0):
    print(s1)
    return render_template("enterreview.html",s1=s1)


def polariy_score(reviw):
    reviewq = reviw
    print(reviewq)
  
 import pandas as pd
    reviewq = pd.Series(reviewq)
    review1 = reviewq.apply(lambda x: " ".join(x.lower() for x in x.split()))
    review1 = review1.str.replace('[^\w\s]', '')
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from textblob import TextBlob
   






 stop = stopwords.words('english')
    # stop = stopwords.words('english')
    review1 = review1.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    st = PorterStemmer()
    review1 = review1.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

    def senti(x):
        return TextBlob(x).sentiment

    rev6 = review1.apply(senti)
    return rev6[0][0]

@app.route("/Moviewnamereviwe",methods=["POST","GET"])
def Moviewnamereviwe():
    if request.method=="POST":
        moviename=request.form["moviewname"]
        review=request.form["review"]
        print(review)
        score=polariy_score(review)
        print(session["username"])
        print(session["username"],review,moviename,score)
        sql="insert into reviews (name,review,moviename,sentiment) values('%s','%s','%s','%s')"%(session["username"],review,moviename,score)
        cursor.execute(sql)
        db.commit()
 sql="insert into history (name,id,history) values('%s','%s','%s')"%(session["username"],session["userid"],moviename)
        cursor.execute(sql)
        db.commit()
        return redirect(url_for('Moviewname'))

@app.route("/View_personal_History")
def View_personal_History():
    sql="select * from history where id='%s'"%(session["userid"])
    data=pd.read_sql_query(sql,db)
    data.drop(["id"],axis=1,inplace=True)
    data["delete"]="delete"
    return render_template("View_personal_History.html",row_val=data.values.tolist())










@app.route("/dlete/<s1>")
def dlete(s1=0):
    print(s1)
    sql="delete from history where sno='%s'"%(s1)
    cursor.execute(sql)
    db.commit()
    return redirect(url_for('View_personal_History'))

@app.route("/View_personal_reviews")
def View_personal_reviews():
    sql="select * from reviews where sno='%s'"%(session["userid"])
    data=pd.read_sql_query(sql,db)
    print(data)
    data.drop(["sno","sentiment","name"],axis=1,inplace=True)
    return render_template("View_personal_reviews.html",row_val=data.values.tolist())
if(__name__)==("__main__"):
    app.run(debug=True)


If no movies found 
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

movies = pd.read_csv("C:/Users/YMTSIN019/Desktop/movies.csv")
ratings = pd.read_csv("C:/Users/YMTSIN019/Desktop/ratings.csv")
final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')

final_dataset.fillna(0,inplace=True)

no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]
final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]

sample = np.array([[0,0,3,0,0],[4,0,0,0,2],[0,0,0,0,1]])
sparsity = 1.0 - ( np.count_nonzero(sample) / float(sample.size) )







csr_sample = csr_matrix(sample)

csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]
   
if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx= final_dataset[final_dataset['movieId']== movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)
        rec_movie_indices=sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        
for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
       
 df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    else:
        return "No movies found. Please check your input"
