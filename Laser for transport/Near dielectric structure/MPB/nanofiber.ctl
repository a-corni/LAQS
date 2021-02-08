; 1. define the geometry
(define-param n1 1.449) ; the waveguide dielectric constant 

(define-param n2 1) ; the surrounding low-dielectric material 

(define-param rad 0.235) ; micro-m 

(define-param X 8) ; micro-m the size of the computational cell in the x direction (power of 2,3,5,7>8)

(define-param Y 8) ; micro-m the size of the computational cell in the y direction (power of 2,3,5,7>8)  

(set! geometry-lattice (make lattice (size X Y no-size))) ; X,Y form the basis for all other 3-vectors in the geometry, and the lattice size determines the size of the primitive cell

(set! default-material (make dielectric (index n2))) ; the mesh is initially only composed of air

(set! geometry(list (make cylinder(center 0 0 0)(radius rad)(height infinity)(axis 0 0 1)(material (make dielectric (index n1)))))) ; generate nanofiber

(set! resolution 256)  ; nbr pixels/a 256; 512

;2. define the output you want  


; 2.1 Output/Skip Output of epsilon
;(output-epsilon) 
;(set! output-epsilon (lambda () (print "JB skipping output-epsilon\n")))


; 2.2 Compute the band structure at k-interp points between kzmin, kzmax 
;(define-param kzmin 0) 

;(define-param kzmax 1) 

;(define-param k-interp 2) 

;(set! k-points (interpolate k-interp(list (vector3 0 0 kzmin) (vector3 0 0 kzmax)))) 


;2.3 ???
;(set-param! num-bands 1) 

;(run display-zparities display-yparities) 


;2.4 Output Efield of first two eigenmodes (the two polarization degenerate fundamental modes HE) 

;(output-efield 1) 

;(output-efield 2) 

;2.5 Output the k-vector for some frequencies 
;(define-param f1 1.173) ; frequency for 852 nm in MPB units micro-m/a

;(define-param f1 (/ 1 0.852347) ) ; probe D2; in micro-m/a

;(define-param f2 (/ 1 0.894593) ) ; probe D1; in micro-m/a

(define-param f3 (/ 1 1.057) ) ; 1057 nm ; in micro-m/a

;(define-param f4 (/ 1 0.780) ) ; 780 nm ; in micro-m/a

(find-k EVEN-Y f3 1 1 (vector3 0 0 1) 1e-6 1.3 f3 (* f3 n1) output-efield output-poynting-z output-tot-pwr get-efield)
(find-k ODD-Y f3 1 1 (vector3 0 0 1) 1e-6 1.3 f3 (* f3 n1) output-efield output-poynting-z output-tot-pwr get-efield)


;(find-k ODD-Y f4 1 1 (vector3 0 0 1) 1e-6 1.3 f4 (* f4 n1) output-efield output-poynting-z output-tot-pwr)
;(find-k EVEN-Y f4 1 1 (vector3 0 0 1) 1e-6 1.3 f4 (* f4 n1) output-efield output-poynting-z output-tot-pwr)

;(find-k ODD-Y f1 1 1 (vector3 0 0 1) 1e-6 1.3 f1 (* f1 n1) display-group-velocities output-efield output-poynting-z output-tot-pwr) ; f1 understood c/a=(a/0.852).c/a = c/0.852 = k 

;(find-k NO-PARITY f2 1 1 (vector3 0 0 1) 1e-4 1.3 f2 (* f2 n1)) 

;(find-k NO-PARITY f3 1 1 (vector3 0 0 1) 1e-4 1.3 f3 (* f3 n1))  

;(find-k NO-PARITY f4 1 1 (vector3 0 0 1) 1e-4 1.3 f4 (* f4 n1))   
