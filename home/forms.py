from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm
from django import forms
from .models import PaperDetails_table
from .models import User
from .models import QuestionTable


class LoginForm(forms.Form):
    username = forms.CharField( 
    widget= forms.TextInput(
      attrs={
        "class": "form-control"
       }
      ) 
    )   
    password = forms.CharField( 
    widget= forms.PasswordInput(
      attrs={
        "class": "form-control"
       }
      ) 
    )

class signUpForm(UserCreationForm):
    username = forms.CharField( 
    widget= forms.TextInput(
      attrs={
        "class": "form-control"
       }
      ) 
    )
    email = forms.CharField( 
    widget= forms.TextInput(
      attrs={
        "class": "form-control"
       }
      ) 
    ) 
    password1 = forms.CharField( 
    widget= forms.PasswordInput(
      attrs={
        "class": "form-control"
       }
      ) 
    )  
    password2 = forms.CharField( 
    widget= forms.PasswordInput(
      attrs={
        "class": "form-control"
       }
      ) 
    ) 
    user_type = forms.ChoiceField(
        choices=[("student", "Student"), ("teacher", "Teacher")],
        widget=forms.RadioSelect(),
        initial="student",  # Set the default option to "student"
        required=True,
    )
    class Meta:
        model = User
        fields = ('username','email','password1','password2','user_type')

















# class QuestionForm(ModelForm):
#     class Meta:
#         model = QuestionTable
#         fields= ['question','answer','mark']

#         widgets={
#             'question':forms.TextInput(attrs={'class':'form-group'}),
#             'answer':forms.TextInput(attrs={'class':'form-group'}),
            
#         }


class QuestionForm(ModelForm):
    class Meta:
        model = QuestionTable
        fields= ['question','answer','mark']

        widgets={
            'question':forms.TextInput(attrs={'class':'form-group'}),
            'answer':forms.TextInput(attrs={'class':'form-group'}),
            'mark':forms.TextInput(attrs={'class':'form-groupLast'})
        }

class PaperDetailsForm(ModelForm):
    class Meta:
        model = PaperDetails_table
        fields= ['institution_Name','subject_Name','level_of_study','date']

        widgets={
            'institution_Name':forms.TextInput(attrs={'class':'form-control'}),
            'subject_Name':forms.TextInput(attrs={'class':'form-control'}),
            'level_of_study':forms.TextInput(attrs={'class':'form-control'}),
            'date':forms.DateInput(attrs={'type':'date', 'class':'form-control'}, format='%Y-%m-%d')

        }