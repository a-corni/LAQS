!%f90 -*- f90 -*-
!Author: Pearu Peterson
!Date:   3 Feb 2002
!$Revision$

python module dvode__user__routines 
    interface dvode_user_interface
       subroutine f(n,t,y,ydot,rpar,ipar)
         integer intent(hide) :: n
         double precision intent(in) :: t
         double precision dimension(n),intent(in,c) :: y
         double precision dimension(n),intent(out,c) :: ydot
         double precision intent(hide) :: rpar
         integer intent(hide) :: ipar
       end subroutine f
       subroutine jac(n,t,y,ml,mu,jac,nrowpd,rpar,ipar)
         integer intent(hide) :: n
         double precision :: t
         double precision dimension(n),intent(c,in) :: y
         integer intent(hide) :: ml,mu
         integer intent(hide):: nrowpd
         double precision intent(out) :: jac(nrowpd, n)
         double precision intent(hide) :: rpar
         integer intent(hide) :: ipar
       end subroutine jac
    end interface
end python module dvode__user__routines

python module zvode__user__routines 
    interface zvode_user_interface
       subroutine f(n,t,y,ydot,rpar,ipar)
         integer intent(hide) :: n
         double precision intent(in) :: t
         double complex dimension(n),intent(in,c) :: y
         double complex dimension(n),intent(out,c) :: ydot
         double precision intent(hide) :: rpar
         integer intent(hide) :: ipar
       end subroutine f
       subroutine jac(n,t,y,ml,mu,jac,nrowpd,rpar,ipar)
         integer intent(hide) :: n
         double precision :: t
         double complex dimension(n),intent(c,in) :: y
         integer intent(hide) :: ml,mu
         integer intent(hide):: nrowpd
         double complex intent(out) :: jac(nrowpd, n)
         double precision intent(hide) :: rpar
         integer intent(hide) :: ipar
       end subroutine jac
    end interface
end python module zvode__user__routines
  
python module vode
    interface
       subroutine dvode(f,jac,neq,y,t,tout,itol,rtol,atol,itask,istate,iopt,rwork,lrw,iwork,liw,mf,rpar,ipar)
         ! y1,t,istate = dvode(f,jac,y0,t0,t1,rtol,atol,itask,istate,rwork,iwork,mf)
         callstatement (*f2py_func)(cb_f_in_dvode__user__routines,&neq,y,&t,&tout,&itol,rtol,atol,&itask,&istate,&iopt,rwork,&lrw,iwork,&liw,cb_jac_in_dvode__user__routines,&mf,&rpar,&ipar)
         use dvode__user__routines
         external f
         external jac
         
         integer intent(hide),depend(y) :: neq = len(y)
         double precision dimension(neq),intent(in,out,copy) :: y
         double precision intent(in,out):: t
         double precision intent(in):: tout
         integer intent(hide),depend(atol) :: itol = (len(atol)<=1 && len(rtol)<=1?1:(len(rtol)<=1?2:(len(atol)<=1?3:4)))
         double precision dimension(*),intent(in),check(len(atol)<&
              &=1||len(atol)>=neq),depend(neq) :: atol
         double precision dimension(*),intent(in),check(len(rtol)<&
              &=1||len(rtol)>=neq),depend(neq) :: rtol
         integer intent(in),check(itask>0 && itask<6) :: itask
         integer intent(in,out),check(istate>0 && istate<4) :: istate
         integer intent(hide) :: iopt = 1
         double precision dimension(lrw),intent(in,cache) :: rwork
         integer intent(hide),check(len(rwork)>=lrw),depend(rwork) :: lrw=len(rwork)
         integer dimension(liw),intent(in,cache) :: iwork
         integer intent(hide),check(len(iwork)>=liw),depend(iwork) :: liw=len(iwork)
         integer intent(in) :: mf
         double precision intent(hide) :: rpar = 0.0
         integer intent(hide) :: ipar = 0
       end subroutine dvode
    end interface

    interface
       subroutine zvode(f,jac,neq,y,t,tout,itol,rtol,atol,itask,istate,iopt,zwork,lzw,rwork,lrw,iwork,liw,mf,rpar,ipar)
         ! y1,t,istate = zvode(f,jac,y0,t0,t1,rtol,atol,itask,istate,rwork,iwork,mf)
         callstatement (*f2py_func)(cb_f_in_zvode__user__routines,&neq,y,&t,&tout,&itol,rtol,atol,&itask,&istate,&iopt,zwork,&lzw,rwork,&lrw,iwork,&liw,cb_jac_in_zvode__user__routines,&mf,&rpar,&ipar)
         use zvode__user__routines
         external f
         external jac
         
         integer intent(hide),depend(y) :: neq = len(y)
         double complex dimension(neq),intent(in,out,copy) :: y
         double precision intent(in,out):: t
         double precision intent(in):: tout
         integer intent(hide),depend(atol) :: itol = (len(atol)<=1 && len(rtol)<=1?1:(len(rtol)<=1?2:(len(atol)<=1?3:4)))
         double precision dimension(*),intent(in),check(len(atol)<&
              &=1||len(atol)>=neq),depend(neq) :: atol
         double precision dimension(*),intent(in),check(len(rtol)<&
              &=1||len(rtol)>=neq),depend(neq) :: rtol
         integer intent(in),check(itask>0 && itask<6) :: itask
         integer intent(in,out),check(istate>0 && istate<4) :: istate
         integer intent(hide) :: iopt = 1
         double complex dimension(lzw),intent(in,cache) :: zwork
         integer intent(hide),check(len(zwork)>=lzw),depend(zwork) :: lzw=len(zwork)
         double precision dimension(lrw),intent(in,cache) :: rwork
         integer intent(hide),check(len(rwork)>=lrw),depend(rwork) :: lrw=len(rwork)
         integer dimension(liw),intent(in,cache) :: iwork
         integer intent(hide),check(len(iwork)>=liw),depend(iwork) :: liw=len(iwork)
         integer intent(in) :: mf
         double precision intent(hide) :: rpar = 0.0
         integer intent(hide) :: ipar = 0
       end subroutine zvode

       ! Fake common block for indicating the integer size
       integer :: intvar
       common /types/ intvar
    end interface
end python module vode