; a = 1 um

;;; gaussian beam function ;;;

; complex constant 
(define-param cfb? #t); complex fields boolean
(set! force-complex-fields? cfb?)
(define i (sqrt -1))

; beam parameters
(define w0 1); 1 um waist
(define-param frq-cen (/ 1 0.938)) ; c/a ; beam frequency
(define lbda (/ 1 frq-cen)); 1 um wavelength

; sub-functions

(define (length-to-number wl); turns wavelength to wavenumber
  (/ (* 2 pi) wl)
  ) 

(define (rayleigh-length waist wl); rayleigh length from waist and wavelength
  (/ (* pi waist waist) wl)
  )

(define (curvature z waist wl); radius of curvature
  (let (
	(zR (rayleigh-length waist wl))
	)
    (if (= z 0)
	0
	(/ z (+ (* z z) (* zR zR)))
	)
    )
  )

(define (waist-away z waist wl); radius away from beam waist
  (let* (
	 (xi (/ z (rayleigh-length waist wl)))
	 )
    (* waist (sqrt (+ 1 (* xi xi))))
    )
  )

(define (gouy-phase z waist wl); gouy phase
  (let* (
	 (xi (/ z (rayleigh-length waist wl)))
	 )
    (atan xi)
    )
  )

(define (vector-radius vec); radius (assume field propagates in z direction)
					; (really in 2d is just y, but for ease of 
					; transition to 3d)
  (let* (				
	 (xx (vector3-x vec))
	 (yy (vector3-y vec))
	 (zz (vector3-z vec))
	 )
    (sqrt (+ (* xx xx) (* zz zz)))
    )
  )

(define (gaussian-beam-vec-fun-1 vec waist wl)
  (let* (
	 (y (vector3-y vec))
	 (r (vector-radius vec))
	 (wy (waist-away y waist wl))
	 (k  (- 0 (length-to-number wl)))
	 (Rinv (curvature y waist wl))
	 (phi (gouy-phase y waist wl))
	 )
    (*
     (/ waist wy)
     (exp (/ (* -1 r r) (* wy wy)))
     (exp (* -1 i (+ (* k y) (* k (/ (* r r Rinv) 2 )) (* -1 phi))))
     ;(/ r wy)
     )
    )
  )
  
(define (gaussian-beam-vec-fun-2 vec waist wl)
  (let* (
	 (y (vector3-y vec))
	 (r (vector-radius vec))
	 (wy (waist-away y waist wl))
	 (k  (length-to-number wl))
	 (Rinv (curvature y waist wl))
	 (phi (gouy-phase y waist wl))
	 )
    (*
     (/ waist wy)
     (exp (/ (* -1 r r) (* wy wy)))
     (exp (* -1 i (+ (* k y) (* k (/ (* r r Rinv) 2 )) (* -1 phi))))
     ;(/ r wy)
     )
    )
)
  
(define-param rad 0.150) ; um ; radius of cylinder

;(define-param frq-cen (/ 1 1.057)) ; c/a ; beam frequency

;(define-param simu-time (/ 80 frq-cen)) ; a/c ; simulation time 160 T
;(define-param pulse-length (* 1000000 simu-time)) ; a/c ; pulse temporal width
;(define-param dfrq (/ 1 (* 2 pulse-length))) ; c/a ; frequency change

;(define frq-min (- (/ 1 1.057) dfrq))
;(define frq-max (+ (/ 1 1.057) dfrq))

;(define wvl-min (/ frq-max))
;(define wvl-max (/ frq-min))


;(define frq-cen (* 0.5 (+ frq-min frq-max)))
;(define nfrq 100)

(define ncylinder 1.449)
(set! geometry (list
                (make cylinder
                  (material (make dielectric (index ncylinder)))
                  (radius rad)
                  (center 0)
				  (height infinity)
				  (axis 0 0 1)
				  )))
;; at least 8 pixels per smallest wavelength, i.e. (floor (/ 8 wvl-min))
			  
(set-param! resolution 64) ; dx = 1/25 um ; dt = 0.5dx/c

(define dpml 1) ; um
(define dair 6) ; um

(define boundary-layers (list (make pml (thickness dpml))))
(set! pml-layers boundary-layers)

(define symm (list (make mirror-sym (direction X) (make mirror-sym (direction Z) (phase -1)))));
;(set! symmetries symm)

(define s (* 2 (+ dpml dair)))
(define cell (make lattice (size s s no-size)))
(set! geometry-lattice cell)

;; (is-integrated? true) necessary for any planewave source extending into PML
(define pw-src-1 (make source
                 (src (make continuous-src (frequency frq-cen) (is-integrated? true)))
                 (center 0 (- (* 0.5 s) dpml) 0)
                 (size (- s (* 2 dpml)) 0  (- s (* 2 dpml)))
                 (component Ez)
				 (amp-func (lambda (v) (gaussian-beam-vec-fun-1 
				(vector3+ v (vector3 0 (- dpml (* 0.5 s)) 0))
				w0 
				lbda
				)
			   ))
			   ))
			   
(define pw-src-2 (make source
                 (src (make continuous-src (frequency frq-cen) (is-integrated? true)))
                 (center 0 (- dpml (* 0.5 s)) 0)
                 (size (- s (* 2 dpml)) 0 (- s (* 2 dpml)))
                 (component Ez)
				 (amp-func (lambda (v) (gaussian-beam-vec-fun-2 
				(vector3+ v (vector3 0 (- (* 0.5 s) dpml) 0))
				w0 
				lbda
				)
			   ))
			   ))
			   
(set! sources (list pw-src-2))

(set! k-point (vector3 0))

(define epsilon (/ 1 25))
(define box-x1 (add-flux frq-cen 0 1
                         (make flux-region (center (- (+ dpml epsilon) (* 0.5 s)) 0 0) (size 0 (-  s (* 2 (+ dpml epsilon))) (- s (* 2 (+ dpml epsilon))) ) ) ))
(define box-x2 (add-flux frq-cen 0 1
                         (make flux-region (center (- (* 0.5 s) (+ dpml epsilon)) 0 0) (size 0 (-  s (* 2 (+ dpml epsilon))) (-  s (* 2 (+ dpml epsilon)))))))
(define box-y1 (add-flux frq-cen 0 1
                         (make flux-region (center 0 (- (+ epsilon dpml) (* 0.5 s)) 0) (size (-  s (* 2 (+ dpml epsilon))) 0 (-  s (* 2 (+ dpml epsilon)))))))
(define box-y3 (add-flux frq-cen 0 1
                         (make flux-region (center 0 0 0) (size (-  s (* 2 (+ dpml epsilon))) 0 (-  s (* 2 (+ dpml epsilon)))))))
(define box-y2 (add-flux frq-cen 0 1
                         (make flux-region (center 0 (- (* 0.5 s) (+ epsilon dpml)) 0) (size (-  s (* 2 (+ dpml epsilon))) 0 (-  s (* 2 (+ dpml epsilon)))))))
;(define box-z1 (add-flux frq-cen 0 1
;                         (make flux-region (center 0 0 (* 0.5 s)) (size (- (- s (* 2 dpml)) (* 2 epsilon)) (- (- s (* 2 dpml)) (* 2 epsilon)) 0))))
;(define box-z2 (add-flux frq-cen 0 1
;                         (make flux-region (center 0 0 (* 0.5 s)) (size (- (- s (* 2 dpml)) (* 2 epsilon)) (- (- s (* 2 dpml)) (* 2 epsilon)) 0))))

(use-output-directory)(run-until 40 (at-beginning output-epsilon )(at-every 2 output-efield)) ;output-efield-z))


(save-flux "box-x1-flux" box-x1)
(save-flux "box-x2-flux" box-x2)
(save-flux "box-y1-flux" box-y1)
(save-flux "box-y2-flux" box-y2)
(save-flux "box-y3-flux" box-y3)

(display-fluxes box-x1)
(display-fluxes box-x2)
(display-fluxes box-y1)
(display-fluxes box-y2)
(display-fluxes box-y3)