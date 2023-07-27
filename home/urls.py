from django.urls import path
from home import views

urlpatterns = [
    path('', views.register, name='register'),
    path('teacherPage', views.teacherPage, name='teacher_Page'),
    path('login', views.login_view, name='login_view'),
    path('logout', views.logout_view, name='logout_view'),
    path('form-submission/', views.form_submission, name='form_submission'),
    path('answer', views.form_data_list, name='form_data_list'),
    path('completed/', views.completed, name='completed'),
    path('submit_answer/', views.submit_answer, name='submit_answer'),
    path('previous_question/', views.previous_question, name='previous_question'),
    path('next_question/', views.next_question, name='next_question'),
    path('score', views.score_table, name='score_table'),
    path('user-data/', views.display_user_data, name='display_user_data'),
    path('user-details/<str:username>/', views.user_detail, name='user_details'),
    path('addQuestion', views.addQuestion, name='addQuestion'),
    path('delete/<int:id>', views.Delete_record, name='delete'),
    path('<int:id>', views.Update_Record, name='update'),
    
    
   
   
  
]
