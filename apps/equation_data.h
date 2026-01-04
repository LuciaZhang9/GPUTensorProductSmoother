#ifndef EQUATION_DATA_H_
#define EQUATION_DATA_H_

#include <deal.II/base/function_lib.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor_function.h>

using namespace dealii;

namespace Stokes
{

  template <int dim>
  class FunctionExtractor : public Function<dim>
  {
  public:
    /**
     * Extracting the vector components c = start,...,end-1 from function @p
     * function_in, which is determined by the half-open range @p range = [start, end).
     */
    FunctionExtractor(const Function<dim>                        *function_in,
                      const std::pair<unsigned int, unsigned int> range)
      : Function<dim>(range.second - range.first)
      , function(function_in)
      , shift(range.first)
    {
      Assert(range.first <= range.second, ExcMessage("Invalid range."));
      Assert(function_in, ExcMessage("function_in is null"));
      AssertIndexRange(range.first, function_in->n_components);
      AssertIndexRange(range.second, function_in->n_components + 1);
    }

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      return function->value(p, shift + component);
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim>  &p,
             const unsigned int component = 0) const override
    {
      return function->gradient(p, shift + component);
    }

    const Function<dim> *function;
    const unsigned int   shift;
  };


  template <int dim, typename VelocityFunction, typename PressureFunction>
  class FunctionMerge : public Function<dim>
  {
  public:
    FunctionMerge()
      : Function<dim>(dim + 1)
    {
      AssertDimension(solution_velocity.n_components, dim);
      AssertDimension(solution_pressure.n_components, 1);
    }

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      if (component < dim)
        return solution_velocity.value(p, component);
      else if (component == dim)
        return solution_pressure.value(p);

      AssertThrow(false, ExcMessage("Invalid component."));
      return 0.;
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim>  &p,
             const unsigned int component = 0) const override
    {
      if (component < dim)
        return solution_velocity.gradient(p, component);
      else if (component == dim)
        return solution_pressure.gradient(p);

      AssertThrow(false, ExcMessage("Invalid component."));
      return Tensor<1, dim>{};
    }

    virtual SymmetricTensor<2, dim>
    hessian(const Point<dim>  &p,
            const unsigned int component = 0) const override
    {
      if (component < dim)
        return solution_velocity.hessian(p, component);
      else if (component == dim)
        return solution_pressure.hessian(p, component);

      AssertThrow(false, ExcMessage("Invalid component."));
      return SymmetricTensor<2, dim>{};
    }

  private:
    VelocityFunction solution_velocity;
    PressureFunction solution_pressure;
  };


  template <int dim, bool is_simplified = true>
  class ManufacturedLoad : public Function<dim>
  {
  public:
    ManufacturedLoad(
      const std::shared_ptr<const Function<dim>> solution_function_in)
      : Function<dim>(dim + 1)
      , solution_function(solution_function_in)
    {
      Assert(solution_function_in->n_components == this->n_components,
             ExcMessage("Mismatching number of components."));
    }

    virtual double
    value(const Point<dim> &xx, const unsigned int component = 0) const override
    {
      constexpr auto pressure_index = dim;
      double         value          = 0.;

      /**
       * The manufactured load associated to the velocity block reads
       *
       *    -2 div E + grad p
       *
       * where E = 1/2 (grad u + [grad u]^T) is the linear strain. The
       * divergence of matrix field E is defined by
       *
       *    (div E)_i = sum_j d/dx_j E_ij.
       *
       * The symmetric gradient E of vector field u reads
       *
       *    E_ij = 1/2 * (d/dx_i u_j + d/dx_j u_i).
       *
       * Combining both, we have
       *
       *    (div E)_i = sum_j 1/2 * (d/dx_j d/dx_i u_j + d^2/dx_j^2 u_i).
       *
       * If @p is_simplified is true we use the simplified Stokes' equations, thus
       * replacing the symmetric gradient by the gradient of u:
       *
       *    -lapl u + grad p
       */
      if (component < pressure_index)
        {
          const auto i      = component;
          double     divE_i = 0.;
          for (auto j = 0U; j < dim; ++j)
            {
              const SymmetricTensor<2, dim> &hessian_of_u_j =
                (solution_function->hessian(xx, j));
              const double Dji_of_u_j =
                is_simplified ? 0. : hessian_of_u_j({i, j});
              const double Djj_of_u_i =
                (solution_function->hessian(xx, i))({j, j});
              /// if simplified divE_i is (0.5 * Lapl u_i)
              divE_i += 0.5 * (Dji_of_u_j + Djj_of_u_i);
            }
          const auto &gradp_i =
            solution_function->gradient(xx, pressure_index)[i];
          value = -2. * divE_i + gradp_i;
        }

      /**
       * The manufactured load associated to the pressure block reads
       *
       *    - div u
       *
       * with u being the velocity field. The load vanishes for a
       * divergence-free velocity.
       */
      else if (component == pressure_index)
        {
          double divu = 0.;
          for (auto j = 0U; j < dim; ++j)
            divu += solution_function->gradient(xx, j)[j];
          value = -divu;
        }

      else
        Assert(false, ExcMessage("Invalid component."));

      return value;
    }

  private:
    std::shared_ptr<const Function<dim>> solution_function;
  };

  template <int dim>
  struct SolutionBase
  {
    static const std::vector<double> polynomial_coefficients;
    static constexpr double          mu     = 0.65;
    static constexpr double          sigma2 = 1. / 3.;
  };

  template <>
  const std::vector<double> SolutionBase<2>::polynomial_coefficients = {
    {0., 0., 1., -2., 1.}};

  template <>
  const std::vector<double> SolutionBase<3>::polynomial_coefficients = {
    {0., 0., 1., -2., 1.}};


  namespace NoSlip
  {
    /**
     * Given the univariate polynomial (@p poly)
     *
     *    q(x) = (x-1)^2 * x^2
     *
     * this class represents the vector curl of
     *
     *    PHI(x,y) = q(x) * q(y)
     *
     * in two dimensions. The roots of p(x) lead to no-slip boundary conditions
     * on the unit cube [0,1]^2. This is the reference solution for the stream
     * function formulation in Kanschat, Sharma '14, thus closely connected to
     * the biharmonic problem of stream functions with clamped boundary
     * conditions.
     */
    template <int dim>
    class SolutionVelocity : public Function<dim>, protected SolutionBase<dim>
    {
      static_assert(dim == 2, "Implemented for two dimensions.");
      using SolutionBase<dim>::polynomial_coefficients;

    public:
      SolutionVelocity()
        : Function<dim>(dim)
        , poly(polynomial_coefficients)
      {}

      virtual double
      value(const Point<dim>  &p,
            const unsigned int component = 0) const override
      {
        AssertIndexRange(component, dim);

        const auto          x = p[0];
        const auto          y = p[1];
        std::vector<double> values_x(2U), values_y(2U);
        poly.value(x, values_x);
        poly.value(y, values_y);
        const auto poly_x  = values_x[0];
        const auto Dpoly_x = values_x[1];
        const auto poly_y  = values_y[0];
        const auto Dpoly_y = values_y[1];

        double val = 0.;
        if (component == 0U)
          val = poly_x * Dpoly_y;
        else if (component == 1U)
          val = -Dpoly_x * poly_y;
        else
          AssertThrow(false, ExcMessage("Invalid component."));
        return val;
      }

      virtual Tensor<1, dim>
      gradient(const Point<dim>  &p,
               const unsigned int component = 0) const override
      {
        AssertIndexRange(component, dim);

        const auto x = p[0];
        const auto y = p[1];

        std::vector<double> values_x(3U), values_y(3U);
        poly.value(x, values_x);
        poly.value(y, values_y);
        const auto poly_x   = values_x[0];
        const auto Dpoly_x  = values_x[1];
        const auto D2poly_x = values_x[2];
        const auto poly_y   = values_y[0];
        const auto Dpoly_y  = values_y[1];
        const auto D2poly_y = values_y[2];

        Tensor<1, dim> grad;
        if (component == 0U)
          {
            grad[0] = Dpoly_x * Dpoly_y;
            grad[1] = poly_x * D2poly_y;
          }
        else if (component == 1U)
          {
            grad[0] = -D2poly_x * poly_y;
            grad[1] = -Dpoly_x * Dpoly_y;
          }
        else
          AssertThrow(false, ExcMessage("Invalid component."));
        return grad;
      }

      virtual SymmetricTensor<2, dim>
      hessian(const Point<dim>  &p,
              const unsigned int component = 0) const override
      {
        AssertIndexRange(component, dim);

        const auto x = p[0];
        const auto y = p[1];

        std::vector<double> values_x(4U), values_y(4U);
        poly.value(x, values_x);
        poly.value(y, values_y);
        const auto poly_x   = values_x[0];
        const auto Dpoly_x  = values_x[1];
        const auto D2poly_x = values_x[2];
        const auto D3poly_x = values_x[3];
        const auto poly_y   = values_y[0];
        const auto Dpoly_y  = values_y[1];
        const auto D2poly_y = values_y[2];
        const auto D3poly_y = values_y[3];

        SymmetricTensor<2, dim> hess;
        if (component == 0U)
          {
            hess[0][0] = D2poly_x * Dpoly_y;
            hess[0][1] = Dpoly_x * D2poly_y;
            hess[1][1] = poly_x * D3poly_y;
          }
        else if (component == 1U)
          {
            hess[0][0] = -D3poly_x * poly_y;
            hess[0][1] = -D2poly_x * Dpoly_y;
            hess[1][1] = -Dpoly_x * D2poly_y;
          }
        else
          AssertThrow(false, ExcMessage("Invalid component."));
        return hess;
      }

    private:
      Polynomials::Polynomial<double> poly;
    };



    template <int dim>
    class SolutionPressure : public Function<dim>
    {
      static_assert(dim == 2, "Implemented for two dimensions.");

    public:
      SolutionPressure()
        : Function<dim>(1)
      {}

      virtual double
      value(const Point<dim>  &p,
            const unsigned int component = 0) const override;

      virtual Tensor<1, dim>
      gradient(const Point<dim>  &p,
               const unsigned int component = 0) const override;

      virtual SymmetricTensor<2, dim>
      hessian(const Point<dim>  &p,
              const unsigned int component = 0) const override;
    };

    template <>
    double
    SolutionPressure<2>::value(const Point<2> &p, const unsigned int) const
    {
      using numbers::PI;
      const double x = p(0);
      const double y = p(1);

      return cos(2. * PI * x) * cos(2. * PI * y);
    }

    template <>
    Tensor<1, 2>
    SolutionPressure<2>::gradient(const Point<2> &p, const unsigned int) const
    {
      using numbers::PI;
      const double x = p(0);
      const double y = p(1);

      Tensor<1, 2> grad;
      {
        grad[0] = -2. * PI * sin(2. * PI * x) * cos(2. * PI * y);
        grad[1] = -2. * PI * cos(2. * PI * x) * sin(2. * PI * y);
      }

      return grad;
    }

    template <>
    SymmetricTensor<2, 2>
    SolutionPressure<2>::hessian(const Point<2> &, const unsigned int) const
    {
      AssertThrow(false, ExcMessage("No need for this functionality..."));
      return SymmetricTensor<2, 2>{};
    }

    template <int dim>
    using Solution =
      FunctionMerge<dim, SolutionVelocity<dim>, SolutionPressure<dim>>;


    /**
     * The explicit definition of the divergence-free load which suits the
     * velocity and pressure reference solutions from above.
     */
    template <int dim>
    class Load : public Function<dim>, protected SolutionBase<dim>
    {
      static_assert(dim == 2, "Implemented for two dimensions.");
      using SolutionBase<dim>::polynomial_coefficients;

    public:
      // TODO init @p poly by the coefficients @p polynomial_coefficients
      Load()
        : Function<dim>(dim + 1)
        , poly(std::vector<double>{{0., 0., 1., -2., 1.}})
      {}

      virtual double
      value(const Point<dim>  &p,
            const unsigned int component = 0) const override
      {
        constexpr auto pressure_index = dim;
        double         value          = 0.;

        const auto x = p[0];
        const auto y = p[1];

        std::vector<double> values_x(4U), values_y(4U);
        poly.value(x, values_x);
        poly.value(y, values_y);
        const auto poly_x   = values_x[0];
        const auto Dpoly_x  = values_x[1];
        const auto D2poly_x = values_x[2];
        const auto D3poly_x = values_x[3];
        const auto poly_y   = values_y[0];
        const auto Dpoly_y  = values_y[1];
        const auto D2poly_y = values_y[2];
        const auto D3poly_y = values_y[3];

        if (component < pressure_index)
          {
            if (component == 0U)
              {
                value = -D2poly_x * Dpoly_y - poly_x * D3poly_y;
              }
            else if (component == 1U)
              {
                value = D3poly_x * poly_y + Dpoly_x * D2poly_y;
              }
            else
              AssertThrow(false, ExcMessage("Invalid component."));
          }

        else if (component == pressure_index)
          {
            value = 0.;
          }

        else
          AssertThrow(false, ExcMessage("Invalid component."));

        return value;
      }

      Polynomials::Polynomial<double> poly;
    };
  } // namespace NoSlip

  namespace NoSlipExp
  {
    namespace Exp
    {

      namespace Common
      {
        int
        binom(int n, int k)
        {
          if (k == 0 || k == n)
            return 1;
          return binom(n - 1, k - 1) + binom(n - 1, k);
        }


        double
        leibniz_rule(const std::vector<double> &derivatives_u,
                     const std::vector<double> &derivatives_v)
        {
          Assert(!derivatives_u.empty(), ExcMessage("No values passed!"));
          AssertDimension(derivatives_u.size(), derivatives_v.size());

          const int n = derivatives_u.size() - 1;

          double sum = 0.;

          for (auto i = 0; i <= n; ++i)
            sum += binom(n, i) * derivatives_u[i] * derivatives_v[n - i];

          return sum;
        }

        /**
         * Represents a Gaussian Bell curve
         *
         *    f(x) = beta * exp(alpha(x))
         *
         * where beta and alpha(x) are defined by the typical parameters mu and
         * sigma^2,
         *
         *    beta = 1 / (2 PI sigma^2)^1/2
         *
         *    alpha(x) = - (x - mu)^2 / (2 sigma^2)
         */
        struct GaussianBell
        {
          static constexpr auto PI = numbers::PI;

          GaussianBell(const double mu_in = 0., const double sigma2_in = 1.)
            : mu(mu_in)
            , sigma2(sigma2_in)
            , beta(1. / std::sqrt(2. * PI * sigma2_in))
            , alpha([&]() {
              /// computes polynomial coefficients:   factor * (x - mu)^2
              const double        factor = -1. / (2. * sigma2_in);
              std::vector<double> coeffs(3U, factor);
              coeffs[0] *= mu_in * mu_in; // x^0
              coeffs[1] *= -2. * mu_in;   // x^1
              coeffs[2] *= 1.;            // x^2
              return coeffs;
            }())
          {}

          void
          fill_derivatives(std::vector<double> &values, const double x) const
          {
            std::vector<double> Dalpha(values.size());
            alpha.value(x, Dalpha);
            const auto alpha_x = Dalpha[0];

            std::vector<double> factors(values.size(), 0.);
            std::vector<double> Dfactors(values.size(), 0.);

            /// Df_prev + f_prev * Da
            const auto &make_f = [&](const unsigned int k) {
              return Dfactors[k - 1] + factors[k - 1] * Dalpha[1];
            };

            /// D2f_prev + Df_prev * Da + f_prev * D2a
            const auto &make_Df = [&](const unsigned int k,
                                      const double       D2f_prev) {
              return D2f_prev + Dfactors[k - 1] * Dalpha[1] +
                     factors[k - 1] * Dalpha[2];
            };

            /// fill factors and Dfactors
            if (values.size() > 0)
              {
                factors[0]  = 1.;
                Dfactors[0] = 0.;
              }
            if (values.size() > 1)
              {
                factors[1]  = make_f(1);      // Da
                Dfactors[1] = make_Df(1, 0.); // D2a
              }
            if (values.size() > 2)
              {
                factors[2]  = make_f(2);             // D2a + Da^2
                Dfactors[2] = make_Df(2, Dalpha[3]); // D3a + 2*D2a*Da
              }
            if (values.size() > 3)
              {
                factors[3]  = make_f(3); // D3a + 3*D2a*Da + Da^3
                Dfactors[3] = make_Df(3,
                                      Dalpha[4] + 2. * Dalpha[3] * Dalpha[1] +
                                        2. * Dalpha[2] * Dalpha[2]);
              }
            if (values.size() > 4)
              {
                factors[4] = make_f(4);
              }
            if (values.size() > 5)
              Assert(false, ExcMessage("TODO ..."));

            /// fill all values queried
            for (auto i = 0U; i < values.size(); ++i)
              values[i] = factors[i] * beta * std::exp(alpha_x);
          }

          const double                          mu;
          const double                          sigma2;
          const double                          beta;
          const Polynomials::Polynomial<double> alpha;
        };
      } // namespace Common

      template <int dim>
      class Solution : public Function<dim>, protected SolutionBase<dim>
      {
        // static_assert(dim == 2, "Implemented for two dimensions.");

      public:
        Solution()
          : Function<dim>(1)
          , poly(SolutionBase<dim>::polynomial_coefficients)
          , bell(SolutionBase<dim>::mu, SolutionBase<dim>::sigma2)
        {}

        virtual double
        value(const Point<dim> &p,
              const unsigned int /*component*/ = 0) const override
        {
          const auto x = p[0];
          const auto y = p[1];

          const double s_x = univariate_value(x);
          const double s_y = univariate_value(y);

          if (dim == 2)
            return s_x * s_y;
          else if (dim == 3)
            {
              const auto   z   = p[2];
              const double s_z = univariate_value(z);

              return s_x * s_y * s_z;
            }
        }

        virtual Tensor<1, dim>
        gradient(const Point<dim> &p,
                 const unsigned int /*component*/ = 0) const override
        {
          const auto x = p[0];
          const auto y = p[1];

          const double s_x  = univariate_value(x);
          const double Ds_x = univariate_first_derivative(x);
          const double s_y  = univariate_value(y);
          const double Ds_y = univariate_first_derivative(y);

          Tensor<1, dim> grad;

          if (dim == 2)
            {
              grad[0] = Ds_x * s_y;
              grad[1] = s_x * Ds_y;

              return grad;
            }
          else if (dim == 3)
            {
              const auto   z    = p[2];
              const double s_z  = univariate_value(z);
              const double Ds_z = univariate_first_derivative(z);

              grad[0] = Ds_x * s_y * s_z;
              grad[1] = s_x * Ds_y * s_z;
              grad[2] = s_x * s_y * Ds_z;

              return grad;
            }
        }

        virtual SymmetricTensor<2, dim>
        hessian(const Point<dim> &p,
                const unsigned int /*component*/ = 0) const override
        {
          const auto x = p[0];
          const auto y = p[1];

          const double s_x   = univariate_value(x);
          const double Ds_x  = univariate_first_derivative(x);
          const double D2s_x = univariate_second_derivative(x);
          const double s_y   = univariate_value(y);
          const double Ds_y  = univariate_first_derivative(y);
          const double D2s_y = univariate_second_derivative(y);

          SymmetricTensor<2, dim> hess;

          if (dim == 2)
            {
              hess[0][0] = D2s_x * s_y;
              hess[0][1] = Ds_x * Ds_y;
              hess[1][1] = s_x * D2s_y;

              return hess;
            }
          else if (dim == 3)
            {
              const auto   z     = p[2];
              const double s_z   = univariate_value(z);
              const double Ds_z  = univariate_first_derivative(z);
              const double D2s_z = univariate_second_derivative(z);

              hess[0][0] = D2s_x * s_y * s_z;
              hess[0][1] = Ds_x * Ds_y * s_z;
              hess[0][2] = Ds_x * s_y * Ds_z;

              hess[1][1] = s_x * D2s_y * s_z;
              hess[1][2] = s_x * Ds_y * Ds_z;

              hess[2][2] = s_x * s_y * D2s_z;

              return hess;
            }
        }

        double
        bilaplacian(const Point<dim> &p,
                    const unsigned int /*component*/ = 0) const
        {
          const auto &x = p[0];
          const auto &y = p[1];

          const double s_x   = univariate_value(x);
          const double D2s_x = univariate_second_derivative(x);
          const double D4s_x = univariate_fourth_derivative(x);
          const double s_y   = univariate_value(y);
          const double D2s_y = univariate_second_derivative(y);
          const double D4s_y = univariate_fourth_derivative(y);

          double bilapl = 0.;
          bilapl        = D4s_x * s_y + 2. * D2s_x * D2s_y + s_x * D4s_y;

          return bilapl;
        }

        void
        fill_derivatives_poly(std::vector<double> &values, const double x) const
        {
          poly.value(x, values);
        }

        void
        fill_derivatives(std::vector<double> &values, const double x) const
        {
          using numbers::PI;

          /// f(x) is Gaussian bell
          bell.fill_derivatives(values, x);


          if (values.size() > 5)
            Assert(false, ExcMessage("TODO ..."));
        }

        double
        make_univariate_derivative(const double x, const unsigned int k) const
        {
          std::vector<double> derivs_poly_x(k + 1);
          fill_derivatives_poly(derivs_poly_x, x);

          std::vector<double> derivs_func_x(k + 1);
          fill_derivatives(derivs_func_x, x);

          return Common::leibniz_rule(derivs_poly_x, derivs_func_x);
        }

        double
        univariate_value(const double x) const
        {
          return make_univariate_derivative(x, 0);
        }

        double
        univariate_first_derivative(const double x) const
        {
          return make_univariate_derivative(x, 1);
        }

        double
        univariate_second_derivative(const double x) const
        {
          return make_univariate_derivative(x, 2);
        }

        double
        univariate_third_derivative(const double x) const
        {
          return make_univariate_derivative(x, 3);
        }

        double
        univariate_fourth_derivative(const double x) const
        {
          return make_univariate_derivative(x, 4);
        }

      private:
        Polynomials::Polynomial<double> poly;
        Common::GaussianBell            bell;
      };
    } // namespace Exp

    /// TODO
    template <int dim>
    class SolutionVelocity : public Function<dim>
    {
      // static_assert(dim == 2, "Implemented for two dimensions.");

    public:
      SolutionVelocity()
        : Function<dim>(dim)
      {}

      virtual double
      value(const Point<dim>  &p,
            const unsigned int component = 0) const override
      {
        AssertIndexRange(component, dim);

        const auto x = p[0];
        const auto y = p[1];

        const auto s_x  = phi.univariate_value(x);
        const auto Ds_x = phi.univariate_first_derivative(x);
        const auto s_y  = phi.univariate_value(y);
        const auto Ds_y = phi.univariate_first_derivative(y);

        if (dim == 2)
          {
            double val = 0.;
            if (component == 0U)
              val = s_x * Ds_y;
            else if (component == 1U)
              val = -Ds_x * s_y;

            return val;
          }
        else if (dim == 3)
          {
            const auto z    = p[2];
            const auto s_z  = phi.univariate_value(z);
            const auto Ds_z = phi.univariate_first_derivative(z);

            double val = 0.;
            if (component == 0U)
              val = s_x * Ds_y * s_z + s_x * s_y * Ds_z;
            else if (component == 1U)
              val = -Ds_x * s_y * s_z - s_x * s_y * Ds_z;
            else if (component == 2U)
              val = -Ds_x * s_y * s_z + s_x * Ds_y * s_z;

            return val;
          }
      }

      virtual Tensor<1, dim>
      gradient(const Point<dim>  &p,
               const unsigned int component = 0) const override
      {
        AssertIndexRange(component, dim);

        const auto x = p[0];
        const auto y = p[1];

        const auto s_x   = phi.univariate_value(x);
        const auto Ds_x  = phi.univariate_first_derivative(x);
        const auto D2s_x = phi.univariate_second_derivative(x);
        const auto s_y   = phi.univariate_value(y);
        const auto Ds_y  = phi.univariate_first_derivative(y);
        const auto D2s_y = phi.univariate_second_derivative(y);

        Tensor<1, dim> grad;
        if (dim == 2)
          {
            if (component == 0U)
              {
                grad[0] = Ds_x * Ds_y;
                grad[1] = s_x * D2s_y;
              }
            else if (component == 1U)
              {
                grad[0] = -D2s_x * s_y;
                grad[1] = -Ds_x * Ds_y;
              }

            return grad;
          }
        else if (dim == 3)
          {
            const auto z     = p[2];
            const auto s_z   = phi.univariate_value(z);
            const auto Ds_z  = phi.univariate_first_derivative(z);
            const auto D2s_z = phi.univariate_second_derivative(z);

            if (component == 0U)
              {
                grad[0] = Ds_x * Ds_y * s_z + Ds_x * s_y * Ds_z;
                grad[1] = s_x * D2s_y * s_z + s_x * Ds_y * Ds_z;
                grad[2] = s_x * Ds_y * Ds_z + s_x * s_y * D2s_z;
              }
            else if (component == 1U)
              {
                grad[0] = -D2s_x * s_y * s_z - Ds_x * s_y * Ds_z;
                grad[1] = -Ds_x * Ds_y * s_z - s_x * Ds_y * Ds_z;
                grad[2] = -Ds_x * s_y * Ds_z - s_x * s_y * D2s_z;
              }
            else if (component == 2U)
              {
                grad[0] = -D2s_x * s_y * s_z + Ds_x * Ds_y * s_z;
                grad[1] = -Ds_x * Ds_y * s_z + s_x * D2s_y * s_z;
                grad[2] = -Ds_x * s_y * Ds_z + s_x * Ds_y * Ds_z;
              }

            return grad;
          }
      }

      virtual SymmetricTensor<2, dim>
      hessian(const Point<dim>  &p,
              const unsigned int component = 0) const override
      {
        AssertIndexRange(component, dim);

        const auto x = p[0];
        const auto y = p[1];

        const auto s_x   = phi.univariate_value(x);
        const auto Ds_x  = phi.univariate_first_derivative(x);
        const auto D2s_x = phi.univariate_second_derivative(x);
        const auto D3s_x = phi.univariate_third_derivative(x);
        const auto s_y   = phi.univariate_value(y);
        const auto Ds_y  = phi.univariate_first_derivative(y);
        const auto D2s_y = phi.univariate_second_derivative(y);
        const auto D3s_y = phi.univariate_third_derivative(y);

        SymmetricTensor<2, dim> hess;

        if (dim == 2)
          {
            if (component == 0U)
              {
                hess[0][0] = D2s_x * Ds_y;
                hess[0][1] = Ds_x * D2s_y;
                hess[1][1] = s_x * D3s_y;
              }
            else if (component == 1U)
              {
                hess[0][0] = -D3s_x * s_y;
                hess[0][1] = -D2s_x * Ds_y;
                hess[1][1] = -Ds_x * D2s_y;
              }

            return hess;
          }
        else if (dim == 3)
          {
            const auto z     = p[2];
            const auto s_z   = phi.univariate_value(z);
            const auto Ds_z  = phi.univariate_first_derivative(z);
            const auto D2s_z = phi.univariate_second_derivative(z);
            const auto D3s_z = phi.univariate_third_derivative(z);

            if (component == 0U)
              {
                hess[0][0] = D2s_x * Ds_y * s_z + D2s_x * s_y * Ds_z;
                hess[0][1] = Ds_x * D2s_y * s_z + Ds_x * Ds_y * Ds_z;
                hess[0][2] = Ds_x * Ds_y * Ds_z + Ds_x * s_y * D2s_z;

                hess[1][1] = s_x * D3s_y * s_z + s_x * D2s_y * Ds_z;
                hess[1][2] = s_x * D2s_y * Ds_z + s_x * Ds_y * D2s_z;

                hess[2][2] = s_x * Ds_y * D2s_z + s_x * s_y * D3s_z;
              }
            else if (component == 1U)
              {
                hess[0][0] = -D3s_x * s_y * s_z - D2s_x * s_y * Ds_z;
                hess[0][1] = -D2s_x * Ds_y * s_z - Ds_x * Ds_y * Ds_z;
                hess[0][2] = -D2s_x * s_y * Ds_z - Ds_x * s_y * D2s_z;

                hess[1][1] = -Ds_x * D2s_y * s_z - s_x * D2s_y * Ds_z;
                hess[1][2] = -Ds_x * Ds_y * Ds_z - s_x * Ds_y * D2s_z;

                hess[2][2] = -Ds_x * s_y * D2s_z - s_x * s_y * D3s_z;
              }
            else if (component == 2U)
              {
                hess[0][0] = -D3s_x * s_y * s_z + D2s_x * Ds_y * s_z;
                hess[0][1] = -D2s_x * Ds_y * s_z + Ds_x * D2s_y * s_z;
                hess[0][2] = -D2s_x * s_y * Ds_z + Ds_x * Ds_y * Ds_z;

                hess[1][1] = -Ds_x * D2s_y * s_z + s_x * D3s_y * s_z;
                hess[1][2] = -Ds_x * Ds_y * Ds_z + s_x * D2s_y * Ds_z;

                hess[2][2] = -Ds_x * s_y * D2s_z + s_x * Ds_y * D2s_z;
              }

            return hess;
          }
      }

    private:
      Exp::Solution<dim> phi;
    };



    /**
     *    p(x,y) = cos(2*PI*x) * cos(2*PI*y) in 2d, and
     *    p(x,y) = cos(2*PI*x) * cos(2*PI*y) * cos(2*PI*z) in 3d.
     */
    template <int dim>
    class SolutionPressure : public Function<dim>
    {
      // static_assert(dim == 2, "Implemented for two dimensions.");

    public:
      SolutionPressure()
        : Function<dim>(1)
      {}

      virtual double
      value(const Point<dim>  &p,
            const unsigned int component = 0) const override;

      virtual Tensor<1, dim>
      gradient(const Point<dim>  &p,
               const unsigned int component = 0) const override;

      virtual SymmetricTensor<2, dim>
      hessian(const Point<dim>  &p,
              const unsigned int component = 0) const override;
    };

    template <>
    double
    SolutionPressure<2>::value(const Point<2> &p, const unsigned int) const
    {
      using numbers::PI;
      const double x = p(0);
      const double y = p(1);

      return cos(2. * PI * x) * cos(2. * PI * y);
    }

    template <>
    Tensor<1, 2>
    SolutionPressure<2>::gradient(const Point<2> &p, const unsigned int) const
    {
      using numbers::PI;
      const double x = p(0);
      const double y = p(1);

      Tensor<1, 2> grad;
      {
        grad[0] = -2. * PI * sin(2. * PI * x) * cos(2. * PI * y);
        grad[1] = -2. * PI * cos(2. * PI * x) * sin(2. * PI * y);
      }

      return grad;
    }

    template <>
    SymmetricTensor<2, 2>
    SolutionPressure<2>::hessian(const Point<2> &, const unsigned int) const
    {
      AssertThrow(false, ExcMessage("No need for this functionality..."));
      return SymmetricTensor<2, 2>{};
    }

    template <>
    double
    SolutionPressure<3>::value(const Point<3> &p, const unsigned int) const
    {
      using numbers::PI;
      const double x = p(0);
      const double y = p(1);
      const double z = p(2);

      return cos(2. * PI * x) * cos(2. * PI * y) * cos(2. * PI * z);
    }

    template <>
    Tensor<1, 3>
    SolutionPressure<3>::gradient(const Point<3> &p, const unsigned int) const
    {
      using numbers::PI;
      const double x = p(0);
      const double y = p(1);
      const double z = p(2);

      Tensor<1, 3> grad;
      {
        grad[0] =
          -2. * PI * sin(2. * PI * x) * cos(2. * PI * y) * cos(2. * PI * z);
        grad[1] =
          -2. * PI * cos(2. * PI * x) * sin(2. * PI * y) * cos(2. * PI * z);
        grad[2] =
          -2. * PI * cos(2. * PI * x) * cos(2. * PI * y) * sin(2. * PI * z);
      }

      return grad;
    }

    template <>
    SymmetricTensor<2, 3>
    SolutionPressure<3>::hessian(const Point<3> &, const unsigned int) const
    {
      AssertThrow(false, ExcMessage("No need for this functionality..."));
      return SymmetricTensor<2, 3>{};
    }

    template <int dim>
    using Solution =
      FunctionMerge<dim, SolutionVelocity<dim>, SolutionPressure<dim>>;
  } // namespace NoSlipExp
} // namespace Stokes



#endif