import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import time
import glob
from io import StringIO
from PIL import Image
import matplotlib.pyplot as plt
from  FlaskObjectDetection.utils  import visualization_utils as vis_util
from  FlaskObjectDetection.utils  import label_map_util
from multiprocessing.dummy import Pool as ThreadPool
import argparse, sys

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import tensorflow._api.v2.compat.v1 as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from flask_bcrypt import Bcrypt
import bcrypt

# Define flask app
app = Flask(__name__, static_url_path='/static')
app.config['IMG_FOLDER'] = 'static/output/'
app.config['IMG_RESIZED_RATIO'] = 500
bcrypt = Bcrypt(app)


# Function to img crop
def crop_img(img, img_name):
    w, h = img.size
    img_resized = None
    if w > h:
        left = (w - h) / 2
        right = left + h
        img_resized = img.crop((left, 0, right, h))
    elif h > w:
        top = (h - w) / 2
        bottom = top + w
        img_resized = img.crop((0, top, w, bottom))
    else:
        img_resized = img

    img_resized = resize_img(img_resized)
    img_resized.save(os.path.join(app.config['IMG_FOLDER'], img_name), 'JPEG', quality=90)

    to_return_path = os.path.join(app.config['IMG_FOLDER'], img_name)

    return to_return_path


# Function to img resized
def resize_img(img):
    size = img.size
    ratio = float(app.config['IMG_RESIZED_RATIO']) / max(size)
    new_img = tuple([int(x * ratio) for x in size])
    return img.resize(new_img, Image.ANTIALIAS)


# Load tensor requirements
MAX_NUMBER_OF_BOXES = 30
MINIMUM_CONFIDENCE = 0.2

PATH_TO_LABELS = 'FlaskObjectDetection/datas/modeloTerminal/label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'datas/img'

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize,
                                                            use_display_name=True)
CATEGORY_INDEX = label_map_util.create_category_index(categories)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#MODEL_NAME = 'FlaskObjectDetection/datas/model'
MODEL_NAME = 'FlaskObjectDetection/datas/modeloTerminal'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# Load model in memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    # Image to numpy array


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Get pest id
def get_pest_id(name):
    sql = "select idpest from pest where name = '{0}'".format(name)
    row = None
    try:
        connection = mysql.connector.connect(host='localhost', database='tesis1', user='root', password='')
        cursor = connection.cursor()
        cursor.execute(sql)
        row = cursor.fetchone()
        print("ID de Plaga:", row[0])
    except Error as error:
        print(error)
    finally:
        cursor.close()
        connection.close()
    return row[0]


# Save history detail
def save_history_detail(idhistory, det_final):
    """-------------------Modified Version---------------------"""
    name = det_final
    idpest = get_pest_id(name)
    sql = "insert into history_detail(idhistory, idpest) values (%s, %s)"
    args = (idhistory, idpest)
    time.sleep(1)
    try:
        connection = mysql.connector.connect(host='localhost', database='tesis1', user='root', password='')
        cursor = connection.cursor()
        cursor.execute(sql, args)
        connection.commit()
    except Error as error:
        print(error)
    finally:
        cursor.close()
        connection.close()


# Save history
def save_history(latitude, longitude, desc, cds, det_final, color_map, id_suer,status_crop):
    sql = "insert into history(latitude, longitude,description,city,history_date,color,iduser,status) values (%s, %s, %s, %s, %s, %s, %s, %s)"
    now = datetime.now()
    args = (latitude, longitude, desc, cds, now.strftime('%y-%m-%d'), color_map, id_suer,status_crop)
    try:
        connection = mysql.connector.connect(host='localhost', database='tesis1', user='root', password='')
        cursor = connection.cursor()
        cursor.execute(sql, args)
        idhistory = cursor.lastrowid
        connection.commit()
        if idhistory:
            if len(det_final) > 0:
                save_history_detail(idhistory, det_final)
        else:
            print("Insert error")

    except Error as error:
        print(error)
    finally:
        cursor.close()
        connection.close()


@app.route('/updt_status', methods=['POST'])
def updt_crop_stat():
    if request.method == 'POST':

        id_history = request.form['id_history']

        try:
            connection = mysql.connector.connect(host='localhost', database='tesis1', user='root', password='')
            cursor = connection.cursor()

            sql = "UPDATE history SET status='Sanitado' where idhistory = '{0}'".format(id_history)
            # args = (id_history)
            cursor.execute(sql)
            connection.commit()
            msgok = "Problema Sanitario Actualizado!!!"
            print(msgok)
            return jsonify(message=msgok)

        except Error as error:
            print(error)
            msgfail = "Error al Actualizar Estado de Plantación!"
            return jsonify(message=msgfail)

        finally:
            connection.close()


# Return pest
"""def get_pest_names(detect):
    pest = []
    for x in detect:
        name = (x[0].split(":")[0] + "s").upper()
        if name == 'ACAROS':
            if len(pest) > 0:
                status = True
                for i in pest:
                    if i == 'ACAROS':
                        status = False
                if status:
                    pest.append(name)
            else:
                pest.append(name)
        elif name == "TRIPS":
            if len(pest) > 0:
                status = True
                for i in pest:
                    if i == "TRIPS":
                        status = False
                if status:
                    pest.append(name)
            else:
                pest.append(name)
    return pest"""


# Get pesticide
def get_pesticide_detect(det_final):
    pest = det_final

    pesticides = []
    size = len(pest)
    if size > 0:
        try:
            connection = mysql.connector.connect(host='localhost', database='tesis1', user='root', password='')
            cursor = connection.cursor()
            """-------------------Modified Version---------------------"""
            sql = "select pesticide.name, img_url,aplication,dose,name_prod,link_prod,url_prodmin from pesticide, pest where pest.idpest = pesticide.idpest and pest.name  = '{0}'".format(
                pest)
            # args = (pest)
            cursor.execute(sql)
            datas = cursor.fetchall()
            # print("Pesticida:",datas[1])

            for r in datas:
                pesticides.append(r)
                # print('Pesticida:',a)
        except Error as error:
            print(error)
        finally:
            cursor.close()
            connection.close()
    return pesticides

    # @app.route('/login', methods=['POST'])
    # def login():
    # msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    """
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:

        username = request.form['username']
        password = request.form['password']

        connection = mysql.connector.connect(host='localhost', database='tesis', user='root', password='moller456')
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes

            msgok= 'Logged in successfully!'
            print(msgok)
            return jsonify(msg=msgok)

        else:
            # Account doesnt exist or username/password incorrect
            msgfail = 'Incorrect username/password!'
            print(msgfail)
            return jsonify(msg=msgfail)"""


@app.route('/login', methods=['POST'])
def login():
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST':

        username = request.form['username']
        password = request.form['password']
        try:
            connection = mysql.connector.connect(host='localhost', database='tesis1', user='root', password='')
            cursor = connection.cursor()
            cursor.execute('SELECT password,name,lastname,Id FROM accounts WHERE username = %s', (username,))
            row = cursor.fetchone()

            if row:
                msgkey = "Usuario Logeado"
                print(row[0])
                nombre = row[1]
                apellido = row[2]
                id_usu = row[3]
                nombcompleto = nombre + " " + apellido

                #   if bcrypt.check_password_hash(row[0],password) ==True:
                if bcrypt.check_password_hash(row[0], password):

                    print(msgkey)
                    msgok = "Usuario Logeado!"
                    return jsonify(message=msgok, nomb_usu=nombcompleto, id_usu=id_usu)

                elif not bcrypt.check_password_hash(row[0], password):

                    msgfail = "Usuario con Contraseña Incorrecta!"
                    print(msgfail)
                    return jsonify(message=msgfail, nomb_usu="", id_usu="")

            else:

                msgfail = "Usuario no se Encuentra Registrado!"
                print(msgfail)
                return jsonify(message=msgfail, nomb_usu="", id_usu="")

        except Error as error:
            print(error)
        finally:
            cursor.close()
            connection.close()


@app.route('/user_register', methods=['POST'])
def register():
    if request.method == 'POST':

        username = request.form['corr']
        password = request.form['clav']
        name = request.form['nomb']
        lastname = request.form['ape']
        telephone = request.form['tel']
        identif = request.form['ced']

        encryptpass = bcrypt.generate_password_hash(password).decode('utf-8')

        try:
            connection = mysql.connector.connect(host='localhost', database='tesis1', user='root', password='')
            cursor = connection.cursor()

            sql = "SELECT * FROM accounts WHERE username = %s"
            cursor.execute(sql, (username,))
            account = cursor.fetchone()
            if account:
                msgexist = "Ya existe este Usuario!"
                print(msgexist)
                return jsonify(message=msgexist)

            else:
                sql = "insert into accounts (username, password, name, lastname, telephone, identification) values (%s, %s, %s, %s, %s, %s)"
                args = (username, encryptpass, name, lastname, telephone, identif)
                cursor.execute(sql, args)
                connection.commit()
                msgok = "Usuario registrado con Éxito!"
                print(msgok)
                return jsonify(message=msgok)

        except Error as error:
            print(error)
            msgfail = "Error al Guardar!"
            return jsonify(message=msgfail)

        finally:
            connection.close()


@app.route('/alarm_register', methods=['POST'])
def reg_alarm():
    if request.method == 'POST':

        desc_alrm = request.form['description']
        time_alrm = request.form['time']
        date_alrm = request.form['date']

        status_alrm = "Programado"
        idbr_alrm = request.form['broadcast']
        user_alrm = request.form['user']

        try:
            connection = mysql.connector.connect(host='localhost', database='tesis1', user='root', password='')
            cursor = connection.cursor()

            sql = "SELECT * FROM alarm_task WHERE time =%s and date =%s and iduser = %s"
            cursor.execute(sql, (time_alrm, date_alrm, user_alrm,))
            account = cursor.fetchone()
            if account:
                msgexist = "Esta alarma ya fue programada!"
                print(msgexist)
                return jsonify(message=msgexist)

            else:

                date_conv = datetime.strptime(date_alrm, "%d-%m-%Y").strftime("%Y-%m-%d")

                date_string = date_conv + " " + time_alrm

                matches = ["a.", "p.", "m."]

                querywords = date_string.split()
                resultwords = [word for word in querywords if word.lower() not in matches]
                dtm_res = ' '.join(resultwords)

                sql = "insert into alarm_task (time,date,description,status,idbroadcast,iduser,datetime_alrm) values (%s, %s, %s, %s, %s, %s, %s)"
                args = (time_alrm, date_alrm, desc_alrm, status_alrm, idbr_alrm, user_alrm, dtm_res)
                cursor.execute(sql, args)
                connection.commit()
                msgok = "Se Guardo!!!"
                print(msgok)
                return jsonify(message=msgok)

        except Error as error:
            print(error)
            msgfail = "Error al Guardar!"
            return jsonify(message=msgfail)

        finally:
            connection.close()



@app.route('/alarm_modify', methods=['POST'])
def mod_alarm():
    if request.method == 'POST':

        desc_alrm = request.form['description']
        time_alrm = request.form['time']
        date_alrm = request.form['date']

        idbr_alrm = request.form['broadcast']
        user_alrm = request.form['user']

        try:
            connection = mysql.connector.connect(host='localhost', database='tesis1', user='root', password='')
            cursor = connection.cursor()

            sql = "SELECT * FROM alarm_task WHERE time =%s and date =%s and iduser = %s"
            cursor.execute(sql, (time_alrm, date_alrm, user_alrm,))
            account = cursor.fetchone()
            if account:
                msgexist = "Esta alarma ya fue programada anteriormente!"
                print(msgexist)
                return jsonify(message=msgexist)

            else:

                date_conv = datetime.strptime(date_alrm, "%d-%m-%Y").strftime("%Y-%m-%d")

                date_string = date_conv + " " + time_alrm

                matches = ["a.", "p.", "m."]

                querywords = date_string.split()
                resultwords = [word for word in querywords if word.lower() not in matches]
                dtm_res = ' '.join(resultwords)
                print("Hora a editar:",time_alrm, date_alrm, desc_alrm, dtm_res, idbr_alrm, user_alrm)

                ##sql = "UPDATE alarm_task SET time=%s, date=%s, description=%s, datetime_alrm=%s  where idbroadcast =%s and iduser =%s"
                ##args = (time_alrm, date_alrm, desc_alrm, dtm_res, idbr_alrm, user_alrm)
                ##cursor.execute(sql, args)
                cursor.execute(""" UPDATE alarm_task SET time=%s, date=%s, description=%s, datetime_alrm=%s  where idbroadcast =%s and iduser =%s  """, (time_alrm, date_alrm, desc_alrm, dtm_res, idbr_alrm, user_alrm))
                connection.commit()
                msgok = "Su actividad fue Editada!!!"
                print(msgok)
                return jsonify(message=msgok)

        except Error as error:
            print(error)
            msgfail = "Error al intentar Editar!"
            return jsonify(message=msgfail)

        finally:
            connection.close()


@app.route('/alarm_cancel', methods=['POST'])
def can_alarm():
    if request.method == 'POST':

        idbr_alrm = request.form['broadcast']
        user_alrm = request.form['user']

        try:
            connection = mysql.connector.connect(host='localhost', database='tesis1', user='root', password='')
            cursor = connection.cursor()

            sql = "UPDATE alarm_task SET status='Cancelado' where idbroadcast =%s and iduser =%s "
            args = (idbr_alrm, user_alrm)
            cursor.execute(sql, args)
            connection.commit()
            msgok = "Se Cancelo la Alarma!!!"
            print(msgok)
            return jsonify(message=msgok)

        except Error as error:
            print(error)
            msgfail = "Error al Cancelar Alarma!"
            return jsonify(message=msgfail)

        finally:
            connection.close()


def get_list(usu):
    markers = None
    # sql = "select idhistory, latitude, longitude, history_date, city from history where history_date = %s and iduser= %s"
    sql = "select time,date,status,description,idbroadcast from alarm_task where alarm_task.status = 'Programado'  and iduser  = '{0}'".format(usu)
    args = (usu)

    try:
        connection = mysql.connector.connect(host='localhost', database='tesis1', user='root', password='')
        cursor = connection.cursor()
        cursor.execute(sql, args)
        datas = cursor.fetchall()
        markers = '['
        for r in datas:
            markers += '{ "time":"' + str(r[0]) + '", "date":"' + str(
                r[1]) + '", "status":"' + str(r[2]) + '", "description":"' + str(r[3]) + '", "idbroadcast":"' + str(
                r[4]) + '"},'
            # en caso de mal funcionamiento ->  + '", "date":"' +  str(r[3]) +
    except Error as error:
        print(error)
    finally:
        cursor.close()
        connection.close()

    markers += ']'
    print(markers)
    return markers


def get_user_position(date):
    markers = None
    # sql = "select idhistory, latitude, longitude, history_date, city from history where history_date = %s and iduser= %s"
    sql = "select history.city,history.history_date,history.latitude,history.longitude ,CONCAT(accounts.name,' ',accounts.lastname) AS people,pest.name " \
          "from accounts,history,history_detail,pest where history.iduser = accounts.Id and history_detail.idhistory = history.idhistory " \
          "and history_detail.idpest = pest.idpest and history.history_date = '{0}'".format(date)
    args = (date)

    try:
        connection = mysql.connector.connect(host='localhost', database='tesis1', user='root', password='')
        cursor = connection.cursor()
        cursor.execute(sql, args)
        datas = cursor.fetchall()
        markers = '['
        for r in datas:
            markers += '{ "city":"' + str(r[0]) + '", "date":"' + str(
                r[1]) + '", "latitude":"' + str(r[2]) + '", "longitude":"' + str(r[3]) + '", "people":"' + str(
                r[4]) + '", "plague":"' + str(r[5]) + '"},'

    except Error as error:
        print(error)
    finally:
        cursor.close()
        connection.close()

    markers += ']'
    print(markers)
    return markers


def get_life_cycle(date):
    life_cycle = []

    chain_received = len(date.split(','))

    if (chain_received >= 2):
        my_list = date.split(",")
        value = tuple(my_list)
        sql = "select pest.name, life_cycle.descripion, population from pest,life_cycle where life_cycle.idpest = pest.idpest and pest.name IN {}".format(
            str(tuple(value)))

    elif (chain_received == 1):
        args = (date)
        sql = "select pest.name, life_cycle.descripion, population from pest,life_cycle where life_cycle.idpest = pest.idpest and pest.name  = '{0}'".format(
            args)

    try:
        connection = mysql.connector.connect(host='localhost', database='tesis1', user='root', password='')
        cursor = connection.cursor()
        cursor.execute(sql)
        datas = cursor.fetchall()

        # life_cycle = '['
        for r in datas:
            # life_cycle  += '{ "name":"' + str(r[0]) + '", "life_cicle":"' + str(r[1]) + '", "population":"' + str(r[2]) + '"},'
            life_cycle.append(r)
    except Error as error:
        print(error)
    finally:
        cursor.close()
        connection.close()

    # life_cycle += ']'
    return life_cycle


# Get marker points
def get_position(date, usu):
    markers = None
    # sql= "select latitude,longitude,name,city from history,history_detail,pest where history_detail.idhistory = history.idhistory and history_detail.idpest = pest.idpest and history_date = %s and iduser = %s"

    sql = "SELECT latitude,longitude,name,city,h.idhistory " \
          "FROM history h " \
          "INNER JOIN history_detail hd ON hd.idhistory = h.idhistory " \
          "INNER JOIN pest p ON hd.idpest =p.idpest " \
          "WHERE history_date = %s and iduser = %s and h.status <> 'Sanitado'" \
          "ORDER BY h.idhistory ASC"

    args = (date, usu)
    print(date)
    try:
        connection = mysql.connector.connect(host='localhost', database='tesis1', user='root', password='')
        cursor = connection.cursor()
        cursor.execute(sql, args)
        datas = cursor.fetchall()
        markers = '['
        for r in datas:
            markers += '{ "latitude":"' + str(r[0]) + '", "longitude":"' + str(
                r[1]) + '", "plague":"' + str(r[2]) + '", "city":"' + str(r[3]) + '", "id":"' + str(r[4]) + '"},'
    except Error as error:
        print(error)
    finally:
        cursor.close()
        connection.close()

    markers += ']'
    print(markers)
    return markers



# IAPP image processing
@app.route('/upload', methods=['POST'])
def img_processing():
    lasted_filename = None
    detection_status = False
    detect = []
    pesticide_solutions = None
   ## pest_names = None
    print("init...")
    #pesticide_solutions = "{" + "}"
    ##pest_names = "{" + "}"
    print("Detecting...")
    if request.method == 'POST':
        # receive img
        f = request.files['image']
        latitude = request.form['lati']
        longitude = request.form['longi']
        desc = request.form['name']

        cds = request.form['ciu']
        id_user = request.form['iduser']
        filename = secure_filename(f.filename)
        request_image = Image.open(f)

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                output_img = Image.open(crop_img(request_image, filename))
                image_np = load_image_into_numpy_array(output_img)
                image_np_expanded = np.expand_dims(image_np, axis=0)

                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                detect = vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    CATEGORY_INDEX,
                    min_score_thresh=MINIMUM_CONFIDENCE,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                im = Image.fromarray(image_np)
                lasted_filename = filename.replace(".jpg", "") + '_output.jpg'
                im.save('static/output/' + lasted_filename)

                detect= [CATEGORY_INDEX.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]
                a = str(detect)[20:-3]
                det_name = (a+ "s").upper()


                json_str = json.dumps(detect)
                resp = json.loads(json_str)
                # print(resp)
                valconv = resp[0]['name']
                #valmayus = (valconv + "s").upper()
                valmayus = (valconv).upper()
                det_final = valmayus
                status_crop = "Riesgo"
                print("Valor convertido:",valmayus)

                if (det_final == "ARACNIDO ROJO"):
                    color_map = "#7BFF01"
                elif (det_final == "PULGON"):
                    color_map = "#02FEFF"
                elif (det_final == "MOSCA BLANCA"):
                     color_map = "#7DFFD2"

        save_history(latitude, longitude, desc, cds, det_final, color_map, id_user,status_crop)

        if len(det_final) > 0:
            detection_status = True

            # I changed these lines from the previous codes to return later a nice and sorted JSON Data,
            # maybe in future version I'm going to add the lifecycle of the proposed plagued in the project

            pesticide_solutions = '['
            for p in get_pesticide_detect(det_final):
                pesticide_solutions += '{ "pest_name":"' + str(p[0]) + '", "img_url":"' + str(
                    p[1]) + '", "aplica":"' + str(p[2]) + '", "dose":"' + str(p[3]) + '", "product_name":"' + str(
                    p[4]) + '", "link":"' + str(p[5]) + '", "url_min":"' + str(p[6]) + '"},'
            pesticide_solutions += ']'

            final_pest_sol = pesticide_solutions.replace('},]', '}]')

            life_cycle = '['
            for lf in get_life_cycle(det_final):
                life_cycle += '{ "name":"' + str(lf[0]) + '", "life_cycle":"' + str(lf[1]) + '", "population":"' + str(
                    lf[2]) + '"},'
            life_cycle += ']'

            final_life_cycle = life_cycle.replace('},]', '}]')

        print("Sin brackets:", det_name)

        print('\n===============================================================================')
        print("Detect:", detect)
        print("Results: ", detection_status)

        print("Pesticida: ", final_pest_sol)
        print('===============================================================================\n')
        print("Ciclo de Vida:", final_life_cycle)
        msg = 'exito'
        return jsonify(message=msg, path=lasted_filename, pest=det_final, pesticide=final_pest_sol, cycle= final_life_cycle)



@app.route('/alrm_list_map', methods=['POST'])
def alrm_listmap():
    if request.method == 'POST':
        print("Peticion....")
        usu = request.form['user']
        return jsonify( get_list(usu).replace('},]', '}]'))


@app.route('/monitoring_list_map', methods=['POST'])
def monitoring_listmap():
    if request.method == 'POST':
        print("Peticion....")
        date = request.form['date']
        return jsonify( get_user_position(date).replace('},]', '}]'))


@app.route('/list_map', methods=['POST'])
def pest_listmap():
    if request.method == 'POST':
        print("Peticion....")
        date = request.form['date']
        usu = request.form['user']
        return jsonify(
            #result=get_position(date).replace('},}', '}}')

            get_position(date,usu).replace('},]', '}]'))


# Run server
if __name__ == '__main__':
    app.run(host="192.168.1.103",port=5000, debug=True)