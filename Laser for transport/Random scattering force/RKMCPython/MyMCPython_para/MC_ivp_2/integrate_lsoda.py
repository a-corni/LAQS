!%f90 -*- f90 -*-
!Author: Pearu Peterson
!Date:   3 Feb 2002
!$Revision$

python module lsoda__user__routines 
    interface lsoda_user_interface
       subroutine f(n,t,y,ydot)
         integer intent(hide) :: n
         double precision intent(in) :: t
         double precision dimension(n),intent(in,c) :: y
         double precision dimension(n),intent(out,c) :: ydot
       end subroutine f
       subroutine jac(n,t,y,ml,mu,jac,nrowpd)
         integer intent(hide) :: n
         double precision :: t
         double precision dimension(n),intent(c,in) :: y
         integer intent(hide) :: ml,mu
         integer intent(hide):: nrowpd
         double precision intent(out) :: jac(nrowpd, n)
       end subroutine jac
    end interface
end python module lsoda__user__routines
  
python module lsoda
    interface
       subroutine lsoda(f,neq,y,t,tout,itol,rtol,atol,itask,istate,iopt,rwork,lrw,iwork,liw,jac,jt)
         ! y1,t,istate = lsoda(f,jac,y0,t0,t1,rtol,atol,itask,istate,rwork,iwork,mf)
         callstatement (*f2py_func)(cb_f_in_lsoda__user__routines,&neq,y,&t,&tout,&itol,rtol,atol,&itask,&istate,&iopt,rwork,&lrw,iwork,&liw,cb_jac_in_lsoda__user__routines,&jt)
         use lsoda__user__routines
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
         integer intent(in) :: jt
       end subroutine lsoda

       ! Fake common block for indicating the integer size
       integer :: intvar
       common /types/ intvar
    end interface
end python module lsoda