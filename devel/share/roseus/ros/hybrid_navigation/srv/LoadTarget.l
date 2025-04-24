;; Auto-generated. Do not edit!


(when (boundp 'hybrid_navigation::LoadTarget)
  (if (not (find-package "HYBRID_NAVIGATION"))
    (make-package "HYBRID_NAVIGATION"))
  (shadow 'LoadTarget (find-package "HYBRID_NAVIGATION")))
(unless (find-package "HYBRID_NAVIGATION::LOADTARGET")
  (make-package "HYBRID_NAVIGATION::LOADTARGET"))
(unless (find-package "HYBRID_NAVIGATION::LOADTARGETREQUEST")
  (make-package "HYBRID_NAVIGATION::LOADTARGETREQUEST"))
(unless (find-package "HYBRID_NAVIGATION::LOADTARGETRESPONSE")
  (make-package "HYBRID_NAVIGATION::LOADTARGETRESPONSE"))

(in-package "ROS")





(defclass hybrid_navigation::LoadTargetRequest
  :super ros::object
  :slots (_image_path ))

(defmethod hybrid_navigation::LoadTargetRequest
  (:init
   (&key
    ((:image_path __image_path) "")
    )
   (send-super :init)
   (setq _image_path (string __image_path))
   self)
  (:image_path
   (&optional __image_path)
   (if __image_path (setq _image_path __image_path)) _image_path)
  (:serialization-length
   ()
   (+
    ;; string _image_path
    4 (length _image_path)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _image_path
       (write-long (length _image_path) s) (princ _image_path s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _image_path
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _image_path (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass hybrid_navigation::LoadTargetResponse
  :super ros::object
  :slots (_success ))

(defmethod hybrid_navigation::LoadTargetResponse
  (:init
   (&key
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _success __success)
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass hybrid_navigation::LoadTarget
  :super ros::object
  :slots ())

(setf (get hybrid_navigation::LoadTarget :md5sum-) "4748f0ca5ea091986703d4cb8411ef83")
(setf (get hybrid_navigation::LoadTarget :datatype-) "hybrid_navigation/LoadTarget")
(setf (get hybrid_navigation::LoadTarget :request) hybrid_navigation::LoadTargetRequest)
(setf (get hybrid_navigation::LoadTarget :response) hybrid_navigation::LoadTargetResponse)

(defmethod hybrid_navigation::LoadTargetRequest
  (:response () (instance hybrid_navigation::LoadTargetResponse :init)))

(setf (get hybrid_navigation::LoadTargetRequest :md5sum-) "4748f0ca5ea091986703d4cb8411ef83")
(setf (get hybrid_navigation::LoadTargetRequest :datatype-) "hybrid_navigation/LoadTargetRequest")
(setf (get hybrid_navigation::LoadTargetRequest :definition-)
      "string image_path
---
bool success 
")

(setf (get hybrid_navigation::LoadTargetResponse :md5sum-) "4748f0ca5ea091986703d4cb8411ef83")
(setf (get hybrid_navigation::LoadTargetResponse :datatype-) "hybrid_navigation/LoadTargetResponse")
(setf (get hybrid_navigation::LoadTargetResponse :definition-)
      "string image_path
---
bool success 
")



(provide :hybrid_navigation/LoadTarget "4748f0ca5ea091986703d4cb8411ef83")


