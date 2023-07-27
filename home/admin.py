from django.contrib import admin
from home.models import scoreOne
from home.models import FData
from home.models import User

# Register your models here.
admin.site.register(User)


@admin.register(scoreOne)
class stu(admin.ModelAdmin):
    list_display = ('user','question','teacher_answer', 'teacher_mark', 'student_answer', 'student_mark')    


@admin.register(FData)
class C(admin.ModelAdmin):
    list_display = ('question', 'answer', 'total_mark','teacher_clusters')




# @admin.register(FormData)
# class FormDataAdmin(admin.ModelAdmin):
#     list_display = ('question', 'answer', 'total_mark', 'display_clusters')

#     def display_clusters(self, obj):
#         return ', '.join([cluster.text for cluster in obj.clusters.all()])

#     display_clusters.short_description = 'Clusters'
