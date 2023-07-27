import json
import matplotlib.pyplot as plt
import numpy as np
#Import all libraries 
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render, redirect
from home.models import Questions, Cluster,FormData,ClusterDB,User,FData,scoreOne
from .forms import signUpForm, LoginForm
from django.contrib.auth import authenticate, login
from django.contrib import messages
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Create your views here.
################# Function to submit questions by teacher #################    
def form_submission(request):
    if request.method == 'POST':
        question = request.POST['question']
        answer = request.POST['answer']
        total_mark = request.POST['totalMark']
        cluster_texts = request.POST.getlist('clusterText[]')
        cluster_marks = request.POST.getlist('clusterMark[]')
        
        if not all(cluster_texts) or not all(cluster_marks):
            error_message = 'Error: Cluster Part was not filled properly!!! You must refill all the textfields again!!!'
            messages.error(request, error_message)
            return render(request, 'addQuestion.html')

        print(answer)
        print('Cluster Texts:', cluster_texts)
        print('Cluster Marks:', cluster_marks)
    
        teacher_clusters = []
        
        for j, cluster_text in enumerate(cluster_texts):
           line_numbers = [int(num) for num in cluster_text.split(',')]
           lines = [line.strip() for i, line in enumerate(answer.split('\n')) if i+1 in line_numbers]
           cluster_label = f"Cluster {j+1}: {lines} (Mark: {cluster_marks[j]})"
           teacher_clusters.append(cluster_label)

        print('\n'.join(teacher_clusters))

        # Create the FormData instance
        form_data = FData.objects.create(question=question, answer=answer, total_mark=total_mark,teacher_clusters=json.dumps(teacher_clusters))
        form_data.save()

    return render(request, 'addQuestion.html')




def form_data_list(request):
    form_data_list = FData.objects.all()
    current_question_index = request.session.get('current_question_index', 0)

    # if current_question_index >= len(form_data_list):
    #     # Redirect to a page indicating that all questions have been answered
    #     return redirect('completed')

    current_question = form_data_list[current_question_index]
   
    context = {
        'question': current_question,
        'question_index': current_question_index + 1,
        'total_questions': len(form_data_list),
    }

    request.session['current_question_index'] = current_question_index

    return render(request, 'answer.html', context)

def completed(request):
    request.session.pop('current_question_index', None)
    return render(request, 'completed.html')

################# Function to submit answer by student - ML algo #################  
def submit_answer(request):
     if request.method == 'POST':
        form_data_list = FData.objects.all()
        current_question_index = request.session.get('current_question_index', 0)
        current_question = form_data_list[current_question_index]       
        student_answer = request.POST.get('answer')

     
     
     print("--------------------------------------------------------------------------------")
     print("                                 Teacher side                                   ")
     print("--------------------------------------------------------------------------------")

     filename = "question.txt"

     # Prompt user for final mark
     Final_mark = current_question.total_mark
     print(f"What mark will you allocate for this question? {Final_mark}")
    #  print("--------------------------------------------------------------------------------")
     # Read the questions and answers from the file
     with open(filename, 'r') as f:
      lines = f.readlines()

     # Replace the question line with current_question.question
     lines[0] = 'question = "{}"\n'.format(current_question.question)

     # Replace the answer lines with current_question.answer
     answers = current_question.answer.split('\n')
     lines[1:] = [answer for answer in answers]

     with open(filename, 'w') as f:
      for line in lines:
        f.write(line)

     # Combine the words within each answer into a single string
     data = []
     for answer in answers:
      words = answer.strip().split(' ')
      data.append(' '.join(words))

     # Vectorize the data
     vectorizer = CountVectorizer()
     X = vectorizer.fit_transform(data)
     
     print("--------------------------------------------------------------------------------")
     print("                                 Teacher cluster                                ")
     print("--------------------------------------------------------------------------------")

     teacher_clustersOne = json.loads(current_question.teacher_clusters)
     print(teacher_clustersOne)


     teacher_clusters = []
     for cluster_string in teacher_clustersOne:
       
       # Extract the cluster mark from the string
       mark_start = cluster_string.find("Mark: ") + len("Mark: ")
       mark_end = cluster_string.find("/", mark_start)
       cluster_mark = float(cluster_string[mark_start:mark_end])

        # Extract the cluster lines from the string
       lines_start = cluster_string.find("['") + len("['")
       lines_end = cluster_string.find("']", lines_start)
       cluster_lines = cluster_string[lines_start:lines_end].split("', '")

       teacher_clusters.append({'cluster': cluster_lines, 'mark': cluster_mark})

     #667E
     filename2 = "student_answer.txt"
     # Read the student answer from the file
     with open(filename2, 'r') as f:
      lines = f.readlines()

     # Replace the current answer lines with new student answer
     answers = request.POST.get('answer').split('\n')
     lines[0:] = [answer for answer in answers]

     with open(filename2, 'w') as f:
      for line in lines:
        f.write(line)
    
     # Open the file
     with open(filename2, "r") as file:
      # Read the contents of the file
      contents = file.readlines()

     # Create an empty list to store the extracted data
     answers = []

     # Iterate over each line in the file contents
     for line in contents:
      # Remove any leading or trailing whitespace
      line = line.strip()
      # Append the line to the answers list
      answers.append(line)

     print("--------------------------------------------------------------------------------")
     print("                                 Student cluster                                ")
     print("--------------------------------------------------------------------------------")

     # Initialize a list to store the matched clusters
     matched_clusters = []
     matched_cluster_indices = []  # Store the indices of the matched clusters

     # Iterate over each line in the student's answer list
     for line in answers:
      # Iterate over the teacher's clusters
      for i, cluster in enumerate(teacher_clusters):
        if line.strip() in cluster['cluster']:
            if i not in matched_cluster_indices:
                # Add the line to the matched cluster
                matched_clusters.append({'cluster': [line.strip()], 'mark': cluster['mark']})
                matched_cluster_indices.append(i)
            else:
                # Add the line to the existing matched cluster
                matched_clusters[matched_cluster_indices.index(i)]['cluster'].append(line.strip())
            break

     # Print the matched clusters
     for cluster in matched_clusters:
      print(f"Matched Cluster: {cluster['cluster']} (Mark: {cluster['mark']:.2f}/{Final_mark:.2f})")

     print("--------------------------------------------------------------------------------")
     print("                                 Similarity algorithm                           ")
     print("--------------------------------------------------------------------------------")

     # Vectorize the clusters in teacher_clusters
     teacher_cluster_vectors = []
     for cluster in teacher_clusters:
      cluster_lines = []
      for line in cluster['cluster']:
          cluster_lines.append(line.strip())
      cluster_text = ' '.join(cluster_lines)

      cluster_vector = vectorizer.transform([cluster_text])
      teacher_cluster_vectors.append(cluster_vector)

     # Vectorize the clusters in matched_clusters
     matched_cluster_vectors = []
     for cluster in matched_clusters:
      cluster_text = ' '.join(cluster['cluster'])
      cluster_vector = vectorizer.transform([cluster_text])
      matched_cluster_vectors.append(cluster_vector)


     # Calculate cosine similarity between each pair of vectors
     similarities = []
     for teacher_vector, matched_vector in zip(teacher_cluster_vectors, matched_cluster_vectors):
       similarity = cosine_similarity(teacher_vector, matched_vector)
       similarities.append(similarity[0][0])
    #111
   

     # Print the cosine similarity results
     print("Cosine Similarity:")
     for i, similarity in enumerate(similarities):
       print(f"Teacher Cluster {i+1} - Matched Cluster {i+1}: {similarity:.4f}")

     #112
     
     #113

     print("--------------------------------------------------------------------------------")
     print("                                  Final Mark                                    ")
     print("--------------------------------------------------------------------------------")


     # Prepare the training data
     X_train = []
     y_train = []

     for cluster in teacher_clusters:
      cluster_lines = ' '.join(cluster['cluster'])
      X_train.append(cluster_lines)
      y_train.append(cluster['mark'])

     print("X_train:")
     print(X_train)

     print("y_train:")
     print(y_train)

     # Vectorize the training data
     vectorizer = CountVectorizer()
     X_train_vectorized = vectorizer.fit_transform(X_train)

     print("X_train_vectorized:")
     print(X_train_vectorized.toarray())

     # Train a linear regression model
     model = LinearRegression()
     model.fit(X_train_vectorized, y_train)

     # Vectorize the student's clusters
     X_test = []
     for cluster in matched_clusters:
      cluster_lines = ' '.join(cluster['cluster'])
      X_test.append(cluster_lines)
     X_test_vectorized = vectorizer.transform(X_test)

     print("X_test:")
     print(X_test)

     print("X_test_vectorized:")
     print(X_test_vectorized.toarray())

     # Predict the marks for the student's clusters
     if len(matched_clusters) > 0:
      y_pred = model.predict(X_test_vectorized)
     else:
        y_pred = []

    #  print("Predicted values (y_pred):")
    #  print(y_pred)   
     
    #  plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted')
    #  plt.plot(range(len(y_pred)), y_pred, color='blue', label='Prediction Line')
    #  plt.title('Predicting marks for each cluster')
    #  plt.xlabel('cluster Index')
    #  plt.ylabel('mark')
    #  plt.show()

     # Calculate the total score based on predicted marks
     total_score = sum([y_pred[i] for i, similarity in enumerate(similarities) if similarity > 0.9])

     # Print the total score
     print("Total Score:", total_score)
     
     #save all the mark for each question in the score table
     question_result = scoreOne(
        user=request.user,
        question=current_question.question,
        teacher_answer=current_question.answer,
        teacher_mark=current_question.total_mark,
        student_answer=student_answer,
        student_mark=total_score
     )
     question_result.save()
    
     request.session['current_question_index'] = current_question_index 

     if current_question_index + 1 >= len(form_data_list):
      return redirect('completed')
     else:            
      return redirect('next_question')  
    
    

################# Function to move to next question #################  
def next_question(request):
    current_question_index = request.session.get('current_question_index', 0)
    form_data_list = FData.objects.all()

    if current_question_index < len(form_data_list) - 1:
        # Move to the next question
        request.session['current_question_index'] += 1

    return redirect('form_data_list')

################# Function to move to previous question ################# 
def previous_question(request):
    current_question_index = request.session.get('current_question_index', 0)

    if current_question_index > 0:
        # Move to the previous question
        request.session['current_question_index'] -= 1

    return redirect('form_data_list')

################# Function for user registration ################# 
def register(request):
    msg=None
    if request.method =='POST':
        form = signUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            msg = 'user created'
            return redirect('login_view')
        else:
            msg = 'form is not valid'
    else:
        form = signUpForm()
    return render(request,'register.html',{'form': form, 'msg': msg })

################# Function for user login ################# 
def login_view(request):
        form = LoginForm(request.POST or None)
        msg = None
        if request.method == 'POST':
            if form.is_valid():
                username = form.cleaned_data.get('username')
                password = form.cleaned_data.get('password')
                user = authenticate(username=username, password=password)
                if user is not None and user.is_student:
                    login(request, user)
                    return redirect('form_data_list')
                elif user is not None and user.is_teacher:
                    login(request, user)
                    return render(request,'teacherPage.html')
                else:
                    msg = 'invalid credentials'
            else:
                msg='error validating form'
        return render(request,'login.html',{'form': form, 'msg': msg })

################# Function for user logout ################# 
def logout_view(request):
  return render(request,'logout.html')

################# Function for score page ################# 
def score_table(request):
    scores = scoreOne.objects.all()
    return render(request, 'scorePage.html', {'scores': scores})

# def ViewscorePage(request):
#   return render(request,'viewScorePage.html')

def display_user_data(request):
    users = scoreOne.objects.values_list('user__username', flat=True).distinct()
    return render(request, 'viewScorePage.html', {'users': users})

def user_detail(request, username):
    user_data = scoreOne.objects.filter(user__username=username)
    return render(request, 'user_detail.html', {'user_data': user_data})

def user_details(request, username):
    user_score = scoreOne.objects.filter(user__username=username).first()

    # if not user_score:
    #     return render(request, 'user_not_found.html')
    return render(request, 'user_details.html', {'user_score': user_score})

def teacherPage(request):
    return render(request, 'teacherPage.html')


#######################################################################################################
 #111
 # # Calculate Euclidean distance between each pair of vectors
    #  distances = []
    #  for teacher_vector, matched_vector in zip(teacher_cluster_vectors, matched_cluster_vectors):
    #   teacher_array = teacher_vector.toarray().flatten()  # Convert to dense array
    #   matched_array = matched_vector.toarray().flatten()  # Convert to dense array
    #   distance = np.linalg.norm(teacher_array - matched_array)
    #   distances.append(distance)

    # # Convert Euclidean distance to similarity score
    #  max_distance = max(distances)  # Maximum possible distance
    #  if max_distance == 0:
    #   similarities = [1.0] * len(distances)  # Set all similarity scores to 1 (maximum similarity)
    #  else:
    #   similarities = [(max_distance - distance) / max_distance for distance in distances]


    #112
    ## Print the similarity results
    #  print("Euclidean Distance (converted to similarity score):")
    #  for i, similarity in enumerate(similarities):
    #   print(f"Teacher Cluster {i+1} - Matched Cluster {i+1}: {similarity:.4f}")

    #113
    ## Plot the cosine similarity values
    #  plt.plot(range(1, len(similarities) + 1), similarities, marker='o')
    #  plt.xlabel('Teacher Cluster')
    #  plt.ylabel('Cosine Similarity')
    #  plt.title('Cosine Similarity between Teacher Clusters and Matched Clusters')
    #  plt.xticks(range(1, len(similarities) + 1))
    #  plt.grid(True)
    #  plt.show()

    #667E
    #  # Prompt the teacher to assign lines to clusters and assign marks to each cluster
    #  teacher_clusters = []
    #  i = 0
    #  while True:
    #   cluster_lines = []
    #   while True:
    #     line_num = input(f"Enter line number(s) for cluster {i+1} (comma-separated), or 'done' to finish: ")
    #     if line_num.lower() == 'done':
    #         break
    #     line_num = line_num.split(',')
    #     for num in line_num:
    #         cluster_lines.append(data[int(num)-1])
    #   if not cluster_lines:
    #     break
    #   cluster_weight = len(cluster_lines) / len(data)
    #   cluster_mark = float(input(f"  Enter mark for cluster {i+1}: "))
    #   teacher_clusters.append({'cluster': ' '.join(cluster_lines), 'mark': cluster_mark})
    #   print(f"  Cluster {i+1}: {cluster_lines} (Contribution: {cluster_weight*100:.2f}%, Mark: {cluster_mark:.2f}/{Final_mark:.2f})")
    #   i += 1


#######################################################################################################

def save_clusters(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        
        for item in data:
            cluster_no = item['cluster_number']
            cluster_lines = item['cluster_line']
            cluster_mark = item['cluster_mark']
            
            Cluster.objects.create(
                clusterNo=cluster_no,
                clusterLines=cluster_lines,
                cluster_mark=cluster_mark
            )

        return JsonResponse({'message': 'Clusters saved successfully!'})


# Delete view
def Delete_record(request):
  return redirect('addQuestion')

# updateView
def Update_Record(request):
    return redirect('update.html')

def addQuestion(request):
    if request.method=="POST":
        question = request.POST['question']
        answer = request.POST['answer']
        mark = request.POST['mark']
        ins = Questions(question=question, answer=answer, mark=mark)
        ins.save()
    return render(request,'addQuestion.html')




