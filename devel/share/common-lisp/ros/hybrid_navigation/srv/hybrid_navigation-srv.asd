
(cl:in-package :asdf)

(defsystem "hybrid_navigation-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "LoadTarget" :depends-on ("_package_LoadTarget"))
    (:file "_package_LoadTarget" :depends-on ("_package"))
  ))