; Auto-generated. Do not edit!


(cl:in-package hybrid_navigation-srv)


;//! \htmlinclude LoadTarget-request.msg.html

(cl:defclass <LoadTarget-request> (roslisp-msg-protocol:ros-message)
  ((image_path
    :reader image_path
    :initarg :image_path
    :type cl:string
    :initform ""))
)

(cl:defclass LoadTarget-request (<LoadTarget-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <LoadTarget-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'LoadTarget-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name hybrid_navigation-srv:<LoadTarget-request> is deprecated: use hybrid_navigation-srv:LoadTarget-request instead.")))

(cl:ensure-generic-function 'image_path-val :lambda-list '(m))
(cl:defmethod image_path-val ((m <LoadTarget-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_navigation-srv:image_path-val is deprecated.  Use hybrid_navigation-srv:image_path instead.")
  (image_path m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <LoadTarget-request>) ostream)
  "Serializes a message object of type '<LoadTarget-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'image_path))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'image_path))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <LoadTarget-request>) istream)
  "Deserializes a message object of type '<LoadTarget-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'image_path) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'image_path) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<LoadTarget-request>)))
  "Returns string type for a service object of type '<LoadTarget-request>"
  "hybrid_navigation/LoadTargetRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'LoadTarget-request)))
  "Returns string type for a service object of type 'LoadTarget-request"
  "hybrid_navigation/LoadTargetRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<LoadTarget-request>)))
  "Returns md5sum for a message object of type '<LoadTarget-request>"
  "4748f0ca5ea091986703d4cb8411ef83")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'LoadTarget-request)))
  "Returns md5sum for a message object of type 'LoadTarget-request"
  "4748f0ca5ea091986703d4cb8411ef83")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<LoadTarget-request>)))
  "Returns full string definition for message of type '<LoadTarget-request>"
  (cl:format cl:nil "string image_path~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'LoadTarget-request)))
  "Returns full string definition for message of type 'LoadTarget-request"
  (cl:format cl:nil "string image_path~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <LoadTarget-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'image_path))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <LoadTarget-request>))
  "Converts a ROS message object to a list"
  (cl:list 'LoadTarget-request
    (cl:cons ':image_path (image_path msg))
))
;//! \htmlinclude LoadTarget-response.msg.html

(cl:defclass <LoadTarget-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass LoadTarget-response (<LoadTarget-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <LoadTarget-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'LoadTarget-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name hybrid_navigation-srv:<LoadTarget-response> is deprecated: use hybrid_navigation-srv:LoadTarget-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <LoadTarget-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_navigation-srv:success-val is deprecated.  Use hybrid_navigation-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <LoadTarget-response>) ostream)
  "Serializes a message object of type '<LoadTarget-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <LoadTarget-response>) istream)
  "Deserializes a message object of type '<LoadTarget-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<LoadTarget-response>)))
  "Returns string type for a service object of type '<LoadTarget-response>"
  "hybrid_navigation/LoadTargetResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'LoadTarget-response)))
  "Returns string type for a service object of type 'LoadTarget-response"
  "hybrid_navigation/LoadTargetResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<LoadTarget-response>)))
  "Returns md5sum for a message object of type '<LoadTarget-response>"
  "4748f0ca5ea091986703d4cb8411ef83")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'LoadTarget-response)))
  "Returns md5sum for a message object of type 'LoadTarget-response"
  "4748f0ca5ea091986703d4cb8411ef83")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<LoadTarget-response>)))
  "Returns full string definition for message of type '<LoadTarget-response>"
  (cl:format cl:nil "bool success ~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'LoadTarget-response)))
  "Returns full string definition for message of type 'LoadTarget-response"
  (cl:format cl:nil "bool success ~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <LoadTarget-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <LoadTarget-response>))
  "Converts a ROS message object to a list"
  (cl:list 'LoadTarget-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'LoadTarget)))
  'LoadTarget-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'LoadTarget)))
  'LoadTarget-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'LoadTarget)))
  "Returns string type for a service object of type '<LoadTarget>"
  "hybrid_navigation/LoadTarget")